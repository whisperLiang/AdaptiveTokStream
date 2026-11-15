from adaptivetokstream.cli import configuration, parser


def test_relative_trace_is_resolved_before_subprocess_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace = tmp_path / "trace"
    trace.write_text("0 1\n1 1\n")
    args = parser().parse_args(
        ["socket-demo", "--asset", "asset", "--rd", "rd", "--trace", "trace", "--output", "out"]
    )
    config = configuration(args)
    assert config.network.trace_path == str(trace.resolve())
