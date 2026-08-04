.PHONY: help setup setup-mac setup-linux install-deps models run gui test lint fmt bench bench-report clean

IN ?=
RUN ?=

help:
	@echo "Targets:"
	@echo "  setup         uv sync (auto-picks mac/linux extras) + subtitler doctor"
	@echo "  install-deps  subtitler doctor --install  (brew on macOS, apt on Debian/Ubuntu)"
	@echo "  models        download the default local model"
	@echo "  run IN=file   subtitler run <file>"
	@echo "  gui           subtitler gui  (opens the browser interface)"
	@echo "  test          pytest"
	@echo "  lint / fmt    ruff check / ruff format"
	@echo "  bench         subtitler bench run"
	@echo "  bench-report RUN=<dir>"

# uname -s is Darwin on macOS, Linux elsewhere. mlx-whisper has no Linux wheels,
# so the extras must differ per platform.
setup:
ifeq ($(shell uname -s),Darwin)
	@$(MAKE) setup-mac
else
	@$(MAKE) setup-linux
endif

setup-mac:
	uv sync --extra mlx --extra cloud --extra dev
	uv run subtitler doctor

setup-linux:
	uv sync --extra local --extra cloud --extra dev
	uv run subtitler doctor

install-deps:
	uv run subtitler doctor --install

models:
	uv run subtitler models download large-v3

run:
	@test -n "$(IN)" || (echo "usage: make run IN=path/to/file" && exit 1)
	uv run subtitler run "$(IN)"

# No extra sync: the GUI is stdlib http.server plus one HTML file, so `make setup` has
# already installed everything it needs.
gui:
	uv run subtitler gui

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

bench:
	uv run subtitler bench run

bench-report:
	@test -n "$(RUN)" || (echo "usage: make bench-report RUN=benchmarks/results/<ts>" && exit 1)
	uv run subtitler bench report "$(RUN)"

clean:
	rm -rf .subtitler build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
