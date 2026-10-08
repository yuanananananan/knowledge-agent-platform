import os

from app import db, init_database, session_scope
from app.models.user import User


def test_standalone_sqlalchemy_supports_model_query_and_nested_scopes(tmp_path):
    original_url = os.getenv("DATABASE_URL")
    sqlite_url = f"sqlite:///{tmp_path / 'runtime.db'}"
    try:
        init_database(sqlite_url)
        db.create_all()
        with session_scope(sqlite_url):
            db.session.add(User(
                id=1,
                userName="runtime-test",
                userAccount="runtime-test",
                userPassword="not-used",
                userRole="user",
                isDelete=False,
            ))
            db.session.commit()
            assert User.query.filter_by(userAccount="runtime-test").one().id == 1

            with session_scope(sqlite_url):
                assert User.query.filter_by(userAccount="runtime-test").count() == 1

            assert User.query.filter_by(userAccount="runtime-test").count() == 1
    finally:
        db.session.remove()
        if original_url:
            init_database(original_url)
