# -*- coding: utf-8 -*-

import copy
import datetime
import functools
import operator
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Type, Union

import pyrql
from sqlalchemy import ColumnElement, Row, RowMapping, exc, func, inspection, orm, sql
from sqlalchemy.orm import attributes, decl_api
from sqlalchemy.sql import _typing, elements, sqltypes

ArgsType = List[Any]
BinaryOperator = Callable[[Any, Any], Any]
SQLAlchemyTypes = Union[
    Type[sqltypes.Integer],
    Type[sqltypes.Float],
    Type[sqltypes.String],
    Type[sqltypes.Boolean],
    Type[sqltypes.DateTime],
    Type[sqltypes.Date],
    Type[sqltypes.LargeBinary],
]


class PaginatedResults(NamedTuple):
    results: Any
    total: int
    previous_page: Optional[str] = None
    next_page: Optional[str] = None


class RQLSelectError(Exception):
    pass


class RQLSelect(sql.Select):
    inherit_cache = True
    _rql_error_cls: Type[Exception] = RQLSelectError

    def __init__(self, *entities: _typing._ColumnsClauseArgument[Any]):
        super().__init__(*entities)
        self._rql_select_clause = []
        self._rql_values_clause = None
        self._rql_scalar_clause = None
        self._rql_where_clause = None
        self._rql_order_by_clause = None
        self._rql_limit_clause = None
        self._rql_offset_clause = None
        self._rql_one_clause = None
        self._rql_distinct_clause = None
        self._rql_group_by_clause = None
        self._rql_joins = []
        self._rql_aliased_models = {}

    @property
    def _rql_select_entities(self) -> List[decl_api.DeclarativeMeta]:
        return [t._annotations["parententity"].entity for t in self._raw_columns]

    @property
    def _rql_select_limit(self) -> int:
        return self._limit_clause.value if self._limit_clause is not None else None

    @property
    def _rql_select_offset(self) -> int:
        return self._offset_clause.value if self._offset_clause is not None else None

    def rql(self, query: str = "", limit: Optional[int] = None) -> sql.Select:
        if len(self._rql_select_entities) > 1:
            raise NotImplementedError("Select must have only one entity")

        if not query:
            self.rql_parsed = None
        else:
            self.rql_expression = query

            try:
                self.rql_parsed: Dict[str, Any] = pyrql.parse(query)
            except pyrql.RQLSyntaxError as e:
                raise self._rql_error_cls(f"RQL Syntax error: {e.args}")

        self._rql_walk(self.rql_parsed)

        select_ = self

        for other in self._rql_joins:
            select_ = select_.outerjoin(*other)

        if self._rql_where_clause is not None:
            select_ = select_.filter(self._rql_where_clause)

        if self._rql_order_by_clause is not None:
            select_ = select_.order_by(*self._rql_order_by_clause)

        if self._rql_limit_clause is not None:
            select_ = select_.limit(self._rql_limit_clause)

        if limit is not None:
            select_ = select_.limit(limit)

        if self._rql_offset_clause is not None:
            select_ = select_.offset(self._rql_offset_clause)

        if self._rql_distinct_clause is not None:
            select_ = select_.distinct()

        return select_

    def rql_all(self, session: orm.Session, is_unique: bool = True) -> Sequence[Union[Union[Row, RowMapping], Any]]:
        """
        Executes the sql expression differently based on which clauses included:
        - For single aggregates a scalar is returned
        - In case the one clause is included only a single row is returned
        - In case a select clause is included only the requisite fields are returned
        - Otherwise (unique) scalars are returned
        """
        if self._rql_scalar_clause is not None:
            return session.scalar(self.with_only_columns(self._rql_scalar_clause))

        if self._rql_one_clause is not None:
            try:
                return [session.scalars(self).one()]
            except exc.NoResultFound:
                raise RQLSelectError("No result found for one()")
            except exc.MultipleResultsFound:
                raise RQLSelectError("Multiple results found for one()")

        if self._rql_values_clause is not None:
            query = self.with_only_columns(self._rql_values_clause)
            if self._rql_distinct_clause is not None:
                query = query.distinct()

            return [row[0] for row in session.execute(query)]

        if self._rql_select_clause:
            query = self.with_only_columns(*self._rql_select_clause)

            if self._rql_group_by_clause:
                query = query.group_by(*self._rql_group_by_clause)

            if self._rql_distinct_clause is not None:
                query = query.distinct()

            return [row._asdict() for row in session.execute(query)]

        if is_unique:
            return session.scalars(self).unique().all()
        return session.scalars(self).all()

    def rql_paginate(self, session: orm.Session) -> PaginatedResults:
        """
        Convenience function for pagination. Returns:
         - the page given to the rql query
         - the count by setting the limit, offset and order by to None
         - next and last page rql queries if more records are available for pagination
        """

        limit = self._rql_select_limit
        offset = self._rql_select_offset or 0

        if limit is None:
            raise RQLSelectError("Pagination requires a limit value")

        page = self.rql_all(session)

        total_query = self.limit(None).offset(None).order_by(None)
        total_query_count = sql.select(func.count()).select_from(total_query.subquery())
        total = session.scalar(total_query_count)

        if offset + limit < total:
            expr = self.rql_expr_replace({"name": "limit", "args": [limit, offset + limit]})
            next_page = expr
        else:
            next_page = None

        if offset > 0 and total:
            expr = self.rql_expr_replace({"name": "limit", "args": [limit, offset - limit]})
            previous_page = expr
        else:
            previous_page = None

        return PaginatedResults(results=page, total=total, previous_page=previous_page, next_page=next_page)

    def rql_expr_replace(self, replacement: Dict[str, Any]) -> str:
        """Replace any nodes matching the replacement name

        This can be used to generate an expression with modified
        `limit` and `offset` nodes, for pagination purposes.

        """
        parsed = copy.deepcopy(self.rql_parsed)

        replaced = self._rql_traverse_and_replace(parsed, replacement["name"], replacement["args"])

        if not replaced:
            parsed = {"name": "and", "args": [replacement, parsed]}

        return pyrql.unparse(parsed)

    def _rql_traverse_and_replace(self, root: Dict[str, Any], name: str, args: ArgsType) -> bool:
        if root is None:
            return False

        if root["name"] == name:
            root["args"] = args
            return True

        else:
            for arg in root["args"]:
                if isinstance(arg, dict):
                    if self._rql_traverse_and_replace(arg, name, args):
                        return True

        return False

    def _rql_walk(self, node: Dict[str, Any]) -> None:
        if node:
            self._rql_where_clause = self._rql_apply(node)

    def _rql_apply(self, node: Dict[str, Any]) -> Any:
        if isinstance(node, dict):
            name = node["name"]
            args = node["args"]

            if name in {"eq", "ne", "lt", "le", "gt", "ge"}:
                return self._rql_compare(args, getattr(operator, name))

            try:
                method = getattr(self, "_rql_" + name)
            except AttributeError:
                raise self._rql_error_cls("Invalid query function: %s" % name)

            return method(args)

        elif isinstance(node, list):
            raise NotImplementedError

        elif isinstance(node, tuple):
            raise NotImplementedError

        return node

    def _rql_attr(
        self, attr: Union[str, Tuple[str, ...]], json_type: SQLAlchemyTypes = sqltypes.Float
    ) -> attributes.InstrumentedAttribute:
        model = self._rql_select_entities[0]

        if isinstance(attr, str):
            return self._get_string_attribute(attr=attr, model=model)

        elif isinstance(attr, tuple):
            root_name = model.__table__.name
            self._rql_aliased_models[root_name] = model
            for i, name in enumerate(attr[:-1]):
                model_path = [root_name, *attr[:i]]
                model_name = "_".join(model_path)
                next_model_path = [root_name, *attr[: i + 1]]
                next_model_name = "_".join(next_model_path)
                model = self._rql_aliased_models[model_name]
                relationships = (
                    inspection.inspect(model).relationships
                    if model_name == root_name
                    else inspection.inspect(inspection.inspect(model).class_).relationships
                )
                if name in relationships:
                    self._handle_relationships(model=model, name=name, next_model_name=next_model_name)
                elif "JSON" in getattr(model, name).type.__class__.__name__.upper():
                    json_attr = self._get_json_attribute(keys=attr[i + 1 :], model=model, name=name)
                    return json_attr.astext.cast(json_type)
                else:
                    raise AttributeError(f'{model} has no relationship or JSON column "{name}"')
            final_path = [root_name, *attr[:-1]]
            final_path_name = "_".join(final_path)
            return getattr(self._rql_aliased_models[final_path_name], attr[-1])

        raise NotImplementedError

    def _rql_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            value = self._rql_apply(value)

        return value

    def _rql_compare(self, args: ArgsType, op: BinaryOperator) -> elements.BinaryExpression:
        attr, value = args
        json_type = _infer_sqlalchemy_type(value)
        attr = self._rql_attr(attr=attr, json_type=json_type)
        value = self._rql_value(value)

        return op(attr, value)

    def _rql_and(self, args: ArgsType) -> Optional[elements.BooleanClauseList]:
        args = [self._rql_apply(node) for node in args]
        args = [a for a in args if a is not None]

        if args:
            return functools.reduce(sql.and_, args)

    def _rql_or(self, args: ArgsType) -> Optional[elements.BooleanClauseList]:
        args = [self._rql_apply(node) for node in args]
        args = [a for a in args if a is not None]

        if args:
            return functools.reduce(sql.or_, args)

    def _rql_in(self, args: ArgsType) -> elements.BinaryExpression:
        attr, value = args
        attr = self._rql_attr(attr=attr, json_type=sqltypes.String)
        value = self._rql_value([str(v) for v in value])

        return attr.in_(value)

    def _rql_out(self, args: ArgsType) -> elements.BinaryExpression:
        attr, value = args
        attr = self._rql_attr(attr=attr, json_type=sqltypes.String)
        value = self._rql_value([str(v) for v in value])

        return sql.not_(attr.in_(value))

    def _rql_like(self, args: ArgsType) -> elements.BinaryExpression:
        attr, value = args
        attr = self._rql_attr(attr=attr, json_type=sqltypes.String)
        value = self._rql_value(value)
        value = value.replace("*", "%")

        return attr.like(value)

    def _rql_limit(self, args: ArgsType) -> None:
        args = [self._rql_value(v) for v in args]

        self._rql_limit_clause = args[0]

        if len(args) == 2:
            self._rql_offset_clause = args[1]

    def _rql_sort(self, args: ArgsType) -> None:
        args = [("+", v) if isinstance(v, str) else v for v in args]
        args = [(p, self._rql_attr(attr=v, json_type=sqltypes.String)) for (p, v) in args]
        attrs = [attr.desc() if p == "-" else attr for (p, attr) in args]

        self._rql_order_by_clause = attrs

    def _rql_contains(self, args: ArgsType) -> ColumnElement[bool]:
        attr, value = args
        json_type = _infer_sqlalchemy_type(value)
        attr = self._rql_attr(attr=attr, json_type=json_type)
        value = self._rql_value(value)

        return attr.contains(value)

    def _rql_excludes(self, args: ArgsType) -> ColumnElement[bool]:
        """
        Take care when excluding values on relationships. It only filters out values where none of the relations
        contain the specified value (see corresponding test).
        """
        attr, value = args
        json_type = _infer_sqlalchemy_type(value)
        attr = self._rql_attr(attr=attr, json_type=json_type)
        value = self._rql_value(value)

        return sql.not_(attr.contains(value))

    def _rql_select(self, args: ArgsType) -> None:
        attrs = [self._rql_attr(attr) for attr in args]

        self._rql_select_clause = attrs

    def _rql_values(self, args: ArgsType) -> None:
        (attr,) = args
        attr = self._rql_attr(attr)

        self._rql_values_clause = attr

    def _rql_distinct(self, *_) -> None:
        self._rql_distinct_clause = True

    def _rql_sum(self, args: ArgsType) -> None:
        (attr,) = args
        attr = self._rql_attr(attr=attr, json_type=sqltypes.Float)
        self._rql_scalar_clause = func.sum(attr)

    def _rql_mean(self, args: ArgsType) -> None:
        (attr,) = args
        attr = self._rql_attr(attr=attr, json_type=sqltypes.Float)

        self._rql_scalar_clause = func.avg(attr)

    def _rql_max(self, args: ArgsType) -> None:
        (attr,) = args
        attr = self._rql_attr(attr=attr, json_type=sqltypes.Float)

        self._rql_scalar_clause = func.max(attr)

    def _rql_min(self, args: ArgsType) -> None:
        (attr,) = args
        attr = self._rql_attr(attr=attr, json_type=sqltypes.Float)

        self._rql_scalar_clause = func.min(attr)

    def _rql_count(self, *_) -> None:
        self._rql_scalar_clause = func.count()

    def _rql_first(self, *_) -> None:
        self._rql_limit_clause = 1

    def _rql_one(self, *_) -> None:
        self._rql_one_clause = True

    def _rql_time(self, args: ArgsType) -> datetime.time:
        return datetime.time(*args)

    def _rql_date(self, args: ArgsType) -> datetime.date:
        return datetime.date(*args)

    def _rql_dt(self, args: ArgsType) -> datetime.datetime:
        return datetime.datetime(*args)

    def _rql_aggregate(self, args: ArgsType) -> None:
        attributes = []
        aggregations = []

        for argument in args:
            if isinstance(argument, dict):
                aggregate_label = argument["name"]
                aggregate_function = getattr(func, argument["name"])
                aggregate_attribute = self._rql_attr(argument["args"][0])

                aggregations.append(aggregate_function(aggregate_attribute).label(aggregate_label))

            else:
                attributes.append(self._rql_attr(argument))

        self._rql_group_by_clause = attributes
        self._rql_select_clause = attributes + aggregations

    def _get_string_attribute(
        self, attr: Union[str, Tuple[str, ...]], model: decl_api.DeclarativeMeta
    ) -> attributes.InstrumentedAttribute:
        try:
            return getattr(model, attr)
        except AttributeError:
            raise self._rql_error_cls("Invalid query attribute: %s" % attr)

    def _get_json_attribute(
        self, keys: List[str], model: Union[orm.util.AliasedClass, decl_api.DeclarativeMeta], name: str
    ):
        json_attr = getattr(model, name)
        for key in keys:
            json_attr = json_attr[key]
        return json_attr

    def _handle_relationships(
        self, model: Union[orm.util.AliasedClass, decl_api.DeclarativeMeta], name: str, next_model_name: str
    ) -> None:
        relation = getattr(model, name)
        relation_model = relation.mapper.class_
        next_model = (
            orm.aliased(relation_model, name=next_model_name)
            if not isinstance(model_ := relation.property.argument, orm.util.AliasedClass)
            else model_
        )
        if next_model_name not in self._rql_aliased_models:
            self._rql_aliased_models[next_model_name] = next_model
            self._rql_joins.append((next_model, relation))


def select(*entities: _typing._ColumnsClauseArgument[Any], **__kw: Any) -> RQLSelect:
    if __kw:
        raise _typing._no_kw()
    return RQLSelect(*entities)


def _infer_sqlalchemy_type(value: Any) -> Type[SQLAlchemyTypes]:
    """
    Infer the most probable SQLAlchemy column type from a given Python value.
    """
    if isinstance(value, int):
        return sqltypes.Integer
    elif isinstance(value, float):
        return sqltypes.Float
    elif isinstance(value, str):
        return sqltypes.String
    elif isinstance(value, bool):
        return sqltypes.Boolean
    elif isinstance(value, datetime.datetime):
        return sqltypes.DateTime
    elif isinstance(value, datetime.date):
        return sqltypes.Date
    elif isinstance(value, bytes):
        return sqltypes.LargeBinary
    elif value is None:
        return sqltypes.String
    else:
        raise ValueError(f"Cannot infer SQLAlchemy type for value of type {type(value)}")

