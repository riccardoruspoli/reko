## [0.4.0] - 2026-09-10

### 🚀 Features

- *(config)* Increase default chunk size to 1200 words
- *(routing)* Add OpenAI direct route selection
- *(workflow)* Add direct OpenAI summary generation

### 🐛 Bug Fixes

- *(workflow)* Report selected model in job metrics
- *(workflow)* Normalize Responses API output
- *(ui)* Order reasoning effort options
- *(prompt)* Enforce direct output language

### 📚 Documentation

- *(readme)* Clarify provider-neutral routing

### 🧪 Testing

- *(config)* Pin the selected chunking default

### ⚙️ Miscellaneous Tasks

- *(docker)* Rotate container logs
## [0.3.1] - 2026-09-06

### 🐛 Bug Fixes

- *(cache)* Persist video titles with transcripts
- *(metadata)* Backfill cached video titles

### ⚙️ Miscellaneous Tasks

- *(deploy)* Improve local deployment defaults
## [0.3.0] - 2026-09-06

### 🚀 Features

- *(transcripts)* Cache raw transcript responses
- *(web)* Hide generation tuning controls
- *(deploy)* Add standalone container deployment
- *(web)* Add transcript refresh toggle
- *(jobs)* Add in-memory job manager
- *(jobs)* Expose asynchronous progress API
- *(web)* Show asynchronous job progress
- *(jobs)* Report cache and phase metrics
- *(web)* Add job retry control

### 🐛 Bug Fixes

- *(web)* Resume SSE after polling fallback
- *(jobs)* Redact sensitive error details
- *(web)* Handle cancelled jobs and metadata failures
- *(web)* Sanitize rendered markdown
- *(compat)* Support Python 3.10 timezones
- *(ci)* Use uv dependabot ecosystem

### 🚜 Refactor

- *(api)* Rename health endpoint
- *(release)* Derive runtime version from package metadata

### 📚 Documentation

- *(release)* Describe publishing prerequisites
- *(web)* Describe asynchronous job lifecycle

### 🎨 Styling

- *(test)* Format test suite

### 🧪 Testing

- Replace live tox checks with pytest
- *(transcripts)* Cover cache refresh behavior
- *(core)* Cover pure processing helpers
- *(adapters)* Cover provider boundaries
- *(generation)* Cover model workflow retries
- *(cli)* Cover command handling
- *(services)* Cover orchestration paths
- *(api)* Cover compatibility and service flows
- Enforce meaningful coverage gate
- *(jobs)* Cover lifecycle and availability
- *(web)* Exercise job progress scenarios
- *(web)* Remove frontend harness

### ⚙️ Miscellaneous Tasks

- *(release)* Add uv build and publishing workflows
- *(test)* Align uv development checks
- *(python)* Pin local interpreter version
- *(test)* Enforce coverage reports
- *(deploy)* Configure job concurrency
## [0.2.0] - 2026-06-08

### 🚀 Features

- Add GPT-5 reasoning model support

### 🐛 Bug Fixes

- Preserve translated key points as separate bullets

### ⚙️ Miscellaneous Tasks

- *(release)* Bump version to 0.2.0
## [0.1.1] - 2025-12-28

### 🚀 Features

- Add local web UI for summarizing YouTube videos

### ⚙️ Miscellaneous Tasks

- Bump version to 0.1.1
## [0.1.0] - 2025-12-22

### 🚀 Features

- Add 'think' mode support for Ollama model
- Add batch processing
- Add summary length option and guidance for video summarization
- Add playlist support for video summarization
- Implement custom error handling and improve argument parsing

### 🐛 Bug Fixes

- Incorrect playlist detection logic
- PyPi naming

### 🚜 Refactor

- Refactor project structure and add new adapters for summarization
- Add subcommands support
- Project structured review
- Enhance models and improve transcript handling
- Method cleanup and import statements update
- Update return types to use specific classes and improve docstrings
- Enhance pyproject.toml with license and metadata
- Improve error handling and logging in CLI

### 📚 Documentation

- Add README.md with project description and features

### ⚙️ Miscellaneous Tasks

- Dependencies cleanup
- Add MIT License
