from qrc_env_template import hello


def test_hello() -> None:
    assert hello() == "Hello from qrc-env-template!"
