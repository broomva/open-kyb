.PHONY: setup lint fmt test examples smoke clean-smoke verify demo witness-drift dogfood docker synth infra-test synth-build sim

DEMO_WS ?= workspaces/kyb-demo
AS_OF ?= 2026-10-25
ASSETS := examples

setup:            ## install the package and dev tools into .venv
	uv sync

lint:             ## ruff lint + format check
	uv run ruff check --no-cache src tests scripts
	uv run ruff format --check --no-cache src tests scripts

fmt:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

test:             ## unit + integration tests
	uv run pytest -q

examples:         ## regenerate synthetic applicant packets under examples/packets/
	uv run python scripts/make_examples.py

smoke: clean-smoke  ## agent-free path: init -> ingest -> pages -> audit verify (packet A)
	uv run kyb init $(DEMO_WS) --tenant kyb-demo --policy $(ASSETS)/policy/compliance_rules.md
	uv run kyb -w $(DEMO_WS) ingest $(ASSETS)/packets/cl-andes-agro --as-of $(AS_OF)
	uv run kyb -w $(DEMO_WS) pages cl-andes-agro
	uv run kyb -w $(DEMO_WS) audit verify
	rm -rf workspaces

clean-smoke:
	rm -rf $(DEMO_WS)

verify:           ## run the evaluation suites (LIVE=1 records witness-eval calls, STRICT=1 fails on pending)
	uv run kyb verify $(if $(filter 1,$(LIVE)),--live) $(if $(filter 1,$(STRICT)),--strict)

LIVE_ENV := env -u GOOGLE_API_KEY KYB_WITNESS_STRATUM=A KYB_WITNESS_A_BACKENDS=vertex-gcp:gemini-3.1-pro-preview,vertex-gcp:gemini-3.8-flash

demo:             ## run end-to-end deterministic review on examples/packets/ and verify audit chain (LIVE=1 runs live harness)
	$(if $(filter 1,$(LIVE)),$(LIVE_ENV)) uv run python scripts/demo.py $(if $(filter 1,$(LIVE)),--live) --workspace $(DEMO_WS)

witness-drift:    ## aggregate recorded witness envelope anomalies (WS=<workspace> or DIR=<dir>)
	uv run python scripts/witness_drift.py $(if $(DIR),--path $(DIR),--ws $(or $(WS),$(DEMO_WS)))

dogfood:          ## run a live agent review on a packet (HARNESS=claude|codex PACKET=...)
	uv run python scripts/dogfood.py --harness $(or $(HARNESS),claude) --packet $(or $(PACKET),es-iberia-software)

IMAGE ?= kyb-service:local

docker:           ## build the service image and verify non-root read-only execution
	@if ! docker info >/dev/null 2>&1; then echo "make docker: SKIPPED, Docker is not available"; exit 0; fi; \
	set -e; \
	docker build -t $(IMAGE) .; \
	docker run --rm --read-only $(IMAGE) --help >/dev/null; \
	uid=$$(docker run --rm --read-only --entrypoint id $(IMAGE) -u); \
	test "$$uid" != 0 || { echo "make docker: FAIL, container runs as root"; exit 1; }; \
	if docker run --rm --read-only --entrypoint sh $(IMAGE) -c 'touch /probe' 2>/dev/null; then \
	  echo "make docker: FAIL, root filesystem is writable"; exit 1; fi; \
	echo "make docker: OK ($(IMAGE) runs --help as uid $$uid on a read-only root filesystem)"

synth:            ## cdk synth the AWS stack into infra/cdk.out
	cd infra && uv run npx -y aws-cdk@2 synth --quiet --no-version-reporting

infra-test:       ## assertions on the synthesized CDK template
	cd infra && uv run pytest -q

SYNTH_N ?= 80
SYNTH_SEED ?= 11

synth-build:      ## build N synthetic KYB cases + rendered packets under ~/.cache/kyb/synth/
	uv run python -m kyb.synth.build --n $(or $(N),$(SYNTH_N)) --seed $(or $(SEED),$(SYNTH_SEED)) $(if $(filter 0,$(RENDER)),--no-render)

sim:              ## autonomy-progression simulation -> artifacts/sim/<RUN>/
	uv run python -m kyb.sim.run --run $(or $(RUN),smoke) --decider $(or $(DECIDER),label) --officer $(or $(OFFICER),sim) --jev $(or $(JEV),replay) --weeks $(or $(WEEKS),8) --k $(or $(K),5) --seed $(or $(SEED),$(SYNTH_SEED)) --n $(or $(N),$(SYNTH_N))
