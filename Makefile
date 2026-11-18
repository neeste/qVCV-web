-include .env
-include ../qCLS/.env

NODE = node
PYTHON = python3
FTP_HOST ?= audres.org
PUBLIC_HOST ?= audres.org
FTP_DIR ?= qVCV

WEB_FILES = index.html style.css qvcv_adaptive.js app.js Stimuli/ltass_noise.wav

all: test

noise: sim/generate_ltass_noise_wav.py
	$(PYTHON) sim/generate_ltass_noise_wav.py

test: qvcv_adaptive.js app.js
	$(NODE) --check qvcv_adaptive.js
	$(NODE) --check app.js
	$(NODE) sim/test_js_adaptive_parity.mjs

# Uploads updated web files + ltass_noise.wav (use DEPLOY_ALL_WAVS=1 for a fresh directory)
deploy:
	@if [ -z "$(FTP_CREDS)" ]; then \
		echo "Error: FTP_CREDS environment variable is not set."; \
		exit 1; \
	fi
	@set -e; for f in $(WEB_FILES); do \
		echo "Uploading $$f -> ftp://$(FTP_HOST)/$(FTP_DIR)/$$f"; \
		curl -sS --ftp-create-dirs -T $$f "ftp://$(FTP_CREDS)@$(FTP_HOST)/$(FTP_DIR)/$$f"; \
	done
	@if [ "$(DEPLOY_ALL_WAVS)" = "1" ]; then \
		set -e; for f in Stimuli/practice/*.wav Stimuli/test/*.wav; do \
			echo "Uploading $$f -> ftp://$(FTP_HOST)/$(FTP_DIR)/$$f"; \
			curl -sS --ftp-create-dirs -T $$f "ftp://$(FTP_CREDS)@$(FTP_HOST)/$(FTP_DIR)/$$f"; \
		done; \
	fi
	@echo "Verify HTTP status:"
	@curl -sS -I "http://$(PUBLIC_HOST)/$(FTP_DIR)/" | head -n 5 || true
