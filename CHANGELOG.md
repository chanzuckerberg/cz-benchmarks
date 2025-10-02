# Changelog

## [0.12.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.11.2...v0.12.0) (2025-09-29)


### Features

* Add Sequential Organization task, metric, and related datasets ([#422](https://github.com/chanzuckerberg/cz-benchmarks/issues/422)) ([f9aa098](https://github.com/chanzuckerberg/cz-benchmarks/commit/f9aa0982431dd133067097e9b70c8efa8062d86c))


### Bug Fixes

* broken links ([#415](https://github.com/chanzuckerberg/cz-benchmarks/issues/415)) ([339974a](https://github.com/chanzuckerberg/cz-benchmarks/commit/339974a109dbe137b7c2c0a4604521dfa7895812))
* doc email link ([#417](https://github.com/chanzuckerberg/cz-benchmarks/issues/417)) ([1880e19](https://github.com/chanzuckerberg/cz-benchmarks/commit/1880e19cf6b3532a952565aa14a6a129b7a5b195))


### Documentation

* rename philosophy page to principles ([#421](https://github.com/chanzuckerberg/cz-benchmarks/issues/421)) ([bfab887](https://github.com/chanzuckerberg/cz-benchmarks/commit/bfab887dbddedceb88f47d2e5f251a4f214b5463))

## [0.11.2](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.11.1...v0.11.2) (2025-09-17)


### Bug Fixes

* for publish-pypi to work on prod pypi ([#413](https://github.com/chanzuckerberg/cz-benchmarks/issues/413)) ([329ac2f](https://github.com/chanzuckerberg/cz-benchmarks/commit/329ac2f23813ee927bb182d092e8ed7cf3b28e1e))

## [0.11.1](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.11.0...v0.11.1) (2025-09-17)


### Bug Fixes

* github actions now use uv properly ([#409](https://github.com/chanzuckerberg/cz-benchmarks/issues/409)) ([01fe9f9](https://github.com/chanzuckerberg/cz-benchmarks/commit/01fe9f95dc176ee5b52b2607a84cf48f739fbbd8))
* tweaking the publish-pypi github action ([#411](https://github.com/chanzuckerberg/cz-benchmarks/issues/411)) ([0307c73](https://github.com/chanzuckerberg/cz-benchmarks/commit/0307c73c7125d16e7bfcdb46399f610868e5e381))

## [0.11.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.10.2...v0.11.0) (2025-09-16)


### Features

* Major API redesign and model removals ([#403](https://github.com/chanzuckerberg/cz-benchmarks/issues/403)) ([51b2a65](https://github.com/chanzuckerberg/cz-benchmarks/commit/51b2a65dd65509d1f5aa3a482d6316fddb4dba10))

## [0.10.2](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.10.1...v0.10.2) (2025-07-18)


### Bug Fixes

* fix UCE docker image build by switching to miniforge due to conda TOS changes ([#333](https://github.com/chanzuckerberg/cz-benchmarks/issues/333)) ([998ddee](https://github.com/chanzuckerberg/cz-benchmarks/commit/998ddee5215ba8a71ee170382d0c160596f902bd))

## [0.10.1](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.10.0...v0.10.1) (2025-07-14)


### Bug Fixes

* scvi hvg alignment ([#290](https://github.com/chanzuckerberg/cz-benchmarks/issues/290)) ([2c90ec8](https://github.com/chanzuckerberg/cz-benchmarks/commit/2c90ec8b2de64072c13190965d0d64b242264fb3))

## [0.10.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.9.2...v0.10.0) (2025-06-16)


### Features

* allow estimating uncertainty in task metrics via batches of random seeds ([#251](https://github.com/chanzuckerberg/cz-benchmarks/issues/251)) ([5c033ff](https://github.com/chanzuckerberg/cz-benchmarks/commit/5c033ffcad31cf66c372c804f9eca91808bd1e19))
* Reapply "geneformer updates" ([#266](https://github.com/chanzuckerberg/cz-benchmarks/issues/266)) ([#267](https://github.com/chanzuckerberg/cz-benchmarks/issues/267)) ([a871b57](https://github.com/chanzuckerberg/cz-benchmarks/commit/a871b57ee3a76bc42603a3fdee2ef00602fe1d98))
* update result output schema ([#252](https://github.com/chanzuckerberg/cz-benchmarks/issues/252)) ([#258](https://github.com/chanzuckerberg/cz-benchmarks/issues/258)) ([8bbb365](https://github.com/chanzuckerberg/cz-benchmarks/commit/8bbb3650acbed9328b9b66a12eafce393990eaa1))


### Bug Fixes

* use macro average for auroc, not weighted ([#253](https://github.com/chanzuckerberg/cz-benchmarks/issues/253)) ([#260](https://github.com/chanzuckerberg/cz-benchmarks/issues/260)) ([412af37](https://github.com/chanzuckerberg/cz-benchmarks/commit/412af37228ffdd868f496ba73f8060ffe0e70787))

## [0.9.2](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.9.1...v0.9.2) (2025-05-20)


### Bug Fixes

* Fix UCE docker image build failure ([#244](https://github.com/chanzuckerberg/cz-benchmarks/issues/244)) ([3acf5fa](https://github.com/chanzuckerberg/cz-benchmarks/commit/3acf5fa4fb031d34dbff378b3e48bad3ef9e1b87))

## [0.9.1](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.9.0...v0.9.1) (2025-05-20)


### Bug Fixes

* Fix transcriptformer Dockerfile ([#224](https://github.com/chanzuckerberg/cz-benchmarks/issues/224)) ([1ebf24a](https://github.com/chanzuckerberg/cz-benchmarks/commit/1ebf24a0b3f5289d1c832606e9918bbef33cbb4a))


### Documentation

* repo and site links update ([#223](https://github.com/chanzuckerberg/cz-benchmarks/issues/223)) ([7c27747](https://github.com/chanzuckerberg/cz-benchmarks/commit/7c27747d37a97bd6288ac1e2cdb38cbec70f17ff))
* tweak roadmap wording ([#217](https://github.com/chanzuckerberg/cz-benchmarks/issues/217)) ([7a6f71d](https://github.com/chanzuckerberg/cz-benchmarks/commit/7a6f71dd5214dd91bc44bd4fdc965aa383bfe029))

## [0.9.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.8.0...v0.9.0) (2025-05-15)


### Features

* add ability to remote cache embeddings and remote upload results ([#207](https://github.com/chanzuckerberg/cz-benchmarks/issues/207)) ([4da7731](https://github.com/chanzuckerberg/cz-benchmarks/commit/4da77311aca958e7bba8d10a3116b11ddc9ccb09))


### Bug Fixes

* fix broken help message ([#216](https://github.com/chanzuckerberg/cz-benchmarks/issues/216)) ([b3f35d0](https://github.com/chanzuckerberg/cz-benchmarks/commit/b3f35d0d9115825182cb96c55e7a6d7801c434c4))
* run_multi_dataset_task needs to know the model args when generating results ([#209](https://github.com/chanzuckerberg/cz-benchmarks/issues/209)) ([2472ebf](https://github.com/chanzuckerberg/cz-benchmarks/commit/2472ebf14fec937441ed570ad1109652cfb6e7aa))


### Documentation

* fix broken links, render warnings ([#212](https://github.com/chanzuckerberg/cz-benchmarks/issues/212)) ([226d4b9](https://github.com/chanzuckerberg/cz-benchmarks/commit/226d4b9ed56380ade7c438011f8629e069b6f4f6))
* may-14 edits from gdocs ([#214](https://github.com/chanzuckerberg/cz-benchmarks/issues/214)) ([ed8279d](https://github.com/chanzuckerberg/cz-benchmarks/commit/ed8279d964290467a52e0e9d43709123d636f80b))
* update roadmap release tags ([#215](https://github.com/chanzuckerberg/cz-benchmarks/issues/215)) ([80d1020](https://github.com/chanzuckerberg/cz-benchmarks/commit/80d1020f6cf88dc637d55fd77ec239e4e8950262))
* VC-2557 && VC-2533 ([#208](https://github.com/chanzuckerberg/cz-benchmarks/issues/208)) ([12c6727](https://github.com/chanzuckerberg/cz-benchmarks/commit/12c6727aeaecf2745321869d9b9608f9884acc87))

## [0.8.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.7.0...v0.8.0) (2025-05-10)


### Features

* aido all model variants ([#175](https://github.com/chanzuckerberg/cz-benchmarks/issues/175)) ([b91ebde](https://github.com/chanzuckerberg/cz-benchmarks/commit/b91ebde908b325d53d508e524cf3f68d73dda038))


### Documentation

* Sync up docs with XFN google docs ([#203](https://github.com/chanzuckerberg/cz-benchmarks/issues/203)) ([6504333](https://github.com/chanzuckerberg/cz-benchmarks/commit/6504333dd46bd7811588b2c9b5535453ced499f4))
* VC-2547 Updated base README ([#199](https://github.com/chanzuckerberg/cz-benchmarks/issues/199)) ([49d0d28](https://github.com/chanzuckerberg/cz-benchmarks/commit/49d0d28ae5aea5fae647bc305bb43ed8de985233))

## [0.7.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.6.1...v0.7.0) (2025-05-08)


### Features

* allow CLI to specify set_baseline for perturbation task ([#178](https://github.com/chanzuckerberg/cz-benchmarks/issues/178)) ([376c833](https://github.com/chanzuckerberg/cz-benchmarks/commit/376c8338a73d6bf2f6ca38589cd87ffbad73bd81))


### Bug Fixes

* Set random_state for RandomForestClassifier ([#197](https://github.com/chanzuckerberg/cz-benchmarks/issues/197)) ([b175388](https://github.com/chanzuckerberg/cz-benchmarks/commit/b17538846fde8433486a713f1e2e2e16f13a4e8e))
* use correct silhouette score ([#187](https://github.com/chanzuckerberg/cz-benchmarks/issues/187)) ([a3e06eb](https://github.com/chanzuckerberg/cz-benchmarks/commit/a3e06eb76eb4eee514411962c8df313b71f11067))


### Documentation

* fixes for PR feedback ([#200](https://github.com/chanzuckerberg/cz-benchmarks/issues/200)) ([8d40da1](https://github.com/chanzuckerberg/cz-benchmarks/commit/8d40da156295c6d330bd71b8d6c3ace90ab2431f))
* improve add custom model docs ([#193](https://github.com/chanzuckerberg/cz-benchmarks/issues/193)) ([7adf963](https://github.com/chanzuckerberg/cz-benchmarks/commit/7adf963a1bc7cb858e9d5895be9b8ad11633ecab))
* VC-2544 Instructions for working on docs ([#192](https://github.com/chanzuckerberg/cz-benchmarks/issues/192)) ([f825c23](https://github.com/chanzuckerberg/cz-benchmarks/commit/f825c230e91469c4ff10c409a3cc42dc6f7b10ff))

## [0.6.1](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.6.0...v0.6.1) (2025-05-02)


### Bug Fixes

* Push transcriptformer docker image to ECR ([#188](https://github.com/chanzuckerberg/cz-benchmarks/issues/188)) ([e5eea6f](https://github.com/chanzuckerberg/cz-benchmarks/commit/e5eea6f025c6bf138a4f2084efb4aea89433317b))

## [0.6.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.5.0...v0.6.0) (2025-05-02)


### Features

* CLI ([#148](https://github.com/chanzuckerberg/cz-benchmarks/issues/148)) ([4cd7947](https://github.com/chanzuckerberg/cz-benchmarks/commit/4cd7947784c5a3dcd351b1d79d191dd52604912d))
* CLI batch jobs ([#161](https://github.com/chanzuckerberg/cz-benchmarks/issues/161)) ([af0bd1a](https://github.com/chanzuckerberg/cz-benchmarks/commit/af0bd1a63c590266946cb5f05cde5d161ebb6d0c))
* enable minimal model developer workflow ([#163](https://github.com/chanzuckerberg/cz-benchmarks/issues/163)) ([f552bf3](https://github.com/chanzuckerberg/cz-benchmarks/commit/f552bf3a45e3622fad21ba4248d4640a2ae49fbf))
* support perturbation in CLI ([#156](https://github.com/chanzuckerberg/cz-benchmarks/issues/156)) ([e14391d](https://github.com/chanzuckerberg/cz-benchmarks/commit/e14391d14dbd8d75f40ff529e1fc7f9efef8ef23))
* transcriptformer ([#165](https://github.com/chanzuckerberg/cz-benchmarks/issues/165)) ([173d1e7](https://github.com/chanzuckerberg/cz-benchmarks/commit/173d1e770631ad436c624a6a9901d98d492942a4))


### Bug Fixes

* Add scGenePT / Perturbation compatibility to developer workflow ([#155](https://github.com/chanzuckerberg/cz-benchmarks/issues/155)) ([feaa4fe](https://github.com/chanzuckerberg/cz-benchmarks/commit/feaa4fe53bcb0000c32507aa277025b2b44f5183))
* Make Geneformer and UCE work in interactive mode again ([#181](https://github.com/chanzuckerberg/cz-benchmarks/issues/181)) ([5128953](https://github.com/chanzuckerberg/cz-benchmarks/commit/512895355a884dc1b0be94c9b8ebafbec3ac3693))
* reload dataset after caching ([#166](https://github.com/chanzuckerberg/cz-benchmarks/issues/166)) ([19ac92b](https://github.com/chanzuckerberg/cz-benchmarks/commit/19ac92b0f1733ba232cabd8816f331a91f83234e))
* use macro, not weighted F1 score (VC-2551) ([#186](https://github.com/chanzuckerberg/cz-benchmarks/issues/186)) ([11dfc05](https://github.com/chanzuckerberg/cz-benchmarks/commit/11dfc0565b9a76e50557826aa22cd5e838b47462))


### Documentation

* docs based on review comments from pr-160 ([#169](https://github.com/chanzuckerberg/cz-benchmarks/issues/169)) ([5e26b84](https://github.com/chanzuckerberg/cz-benchmarks/commit/5e26b84b8a3f2753c104b7ddb226d50ff0949b79))
* Nc docs structure ([#158](https://github.com/chanzuckerberg/cz-benchmarks/issues/158)) ([deaee6f](https://github.com/chanzuckerberg/cz-benchmarks/commit/deaee6f2aeb18f6c85044a217eb19adbb80424be))

## [0.5.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.4.4...v0.5.0) (2025-04-03)


### Features

* random seed for clustering ([#151](https://github.com/chanzuckerberg/cz-benchmarks/issues/151)) ([a1df9f9](https://github.com/chanzuckerberg/cz-benchmarks/commit/a1df9f901125132d2702545a07dea18a99562e80))


### Bug Fixes

* add descriptive comment for pipeline.py ([#110](https://github.com/chanzuckerberg/cz-benchmarks/issues/110)) ([7e0ead3](https://github.com/chanzuckerberg/cz-benchmarks/commit/7e0ead397ed0cfb17d478d76e4f2a0be4aa83dd6))
* developer workflow implementation ([#126](https://github.com/chanzuckerberg/cz-benchmarks/issues/126)) ([703b5eb](https://github.com/chanzuckerberg/cz-benchmarks/commit/703b5eba3ba77c8f9a94eeb34942a56b00f8f6e1))
* remove gene prefix validation from condition keys of perturbation single cell datasets ([#152](https://github.com/chanzuckerberg/cz-benchmarks/issues/152)) ([e766946](https://github.com/chanzuckerberg/cz-benchmarks/commit/e766946d10d1878f307f87bd23056544dc56e465))

## [0.4.4](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.4.3...v0.4.4) (2025-04-01)


### Bug Fixes

* add tag to mount dir ([#144](https://github.com/chanzuckerberg/cz-benchmarks/issues/144)) ([b9d2500](https://github.com/chanzuckerberg/cz-benchmarks/commit/b9d250047f589d2321508360025d6c7495f86c10))
* push model images to ECR public repo ([#133](https://github.com/chanzuckerberg/cz-benchmarks/issues/133)) ([8b5b653](https://github.com/chanzuckerberg/cz-benchmarks/commit/8b5b65357212fc652f03596baf8711b9a0b68c70))

## [0.4.3](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.4.2...v0.4.3) (2025-03-31)


### Bug Fixes

* remove old conf line in docker ([#142](https://github.com/chanzuckerberg/cz-benchmarks/issues/142)) ([69e637a](https://github.com/chanzuckerberg/cz-benchmarks/commit/69e637a3545deba78b53cff740025a0a9e621d81))

## [0.4.2](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.4.1...v0.4.2) (2025-03-31)


### Bug Fixes

* include conf dir in package structure ([#140](https://github.com/chanzuckerberg/cz-benchmarks/issues/140)) ([24e02f6](https://github.com/chanzuckerberg/cz-benchmarks/commit/24e02f6e71559241dc05b19ec4d0e3f5012cb00d))

## [0.4.1](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.4.0...v0.4.1) (2025-03-27)


### Bug Fixes

* add conf directory to all containers ([#128](https://github.com/chanzuckerberg/cz-benchmarks/issues/128)) ([9415e3c](https://github.com/chanzuckerberg/cz-benchmarks/commit/9415e3c63bc8bcf93b9d1e85b4dd57d6532c90d4))

## [0.4.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.3.0...v0.4.0) (2025-03-26)


### Features

* docker builder ([#124](https://github.com/chanzuckerberg/cz-benchmarks/issues/124)) ([6c0e5b5](https://github.com/chanzuckerberg/cz-benchmarks/commit/6c0e5b5cbcc4eadbda971cd630e1e2040d86fe4f))
* move model validators back into docker images ([#120](https://github.com/chanzuckerberg/cz-benchmarks/issues/120)) ([ad1cb59](https://github.com/chanzuckerberg/cz-benchmarks/commit/ad1cb5933d4ced62cfabd84d66370a99ea12f9fc))


### Documentation

* autogenerate documentation - sphinx ([#116](https://github.com/chanzuckerberg/cz-benchmarks/issues/116)) ([94be611](https://github.com/chanzuckerberg/cz-benchmarks/commit/94be611d4eb31f3b69d3d9423299abba9eee3be4))

## [0.3.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.2.2...v0.3.0) (2025-03-19)


### Features

* add back interactive mode and file mounting for container debugging ([#114](https://github.com/chanzuckerberg/cz-benchmarks/issues/114)) ([ba317e5](https://github.com/chanzuckerberg/cz-benchmarks/commit/ba317e5952bfff8a4502788e5b5d0d48bb4ed086))


### Bug Fixes

* move metrics readme to the correct dir ([#112](https://github.com/chanzuckerberg/cz-benchmarks/issues/112)) ([1dcd620](https://github.com/chanzuckerberg/cz-benchmarks/commit/1dcd6209617e1ccf06bcfdb1542871bdf55440d0))

## [0.2.2](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.2.1...v0.2.2) (2025-03-18)


### Bug Fixes

* geneformer model updates ([#103](https://github.com/chanzuckerberg/cz-benchmarks/issues/103)) ([5fdfdff](https://github.com/chanzuckerberg/cz-benchmarks/commit/5fdfdffa326c0f29fccc5fe89311cdefc7bc6e08))

## [0.2.1](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.2.0...v0.2.1) (2025-03-13)


### Bug Fixes

* test release ([#98](https://github.com/chanzuckerberg/cz-benchmarks/issues/98)) ([d710766](https://github.com/chanzuckerberg/cz-benchmarks/commit/d7107664203d12c144d928176971a8ca9a542ba3))

## [0.2.0](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.1.2...v0.2.0) (2025-03-13)


### Features

* perturb baselines ([#86](https://github.com/chanzuckerberg/cz-benchmarks/issues/86)) ([9fe91be](https://github.com/chanzuckerberg/cz-benchmarks/commit/9fe91be296396530af3bb668d2531af94a8174a5))


### Bug Fixes

* Improve error message for var_names and SCVI dataset validation ([#83](https://github.com/chanzuckerberg/cz-benchmarks/issues/83)) ([ac048b0](https://github.com/chanzuckerberg/cz-benchmarks/commit/ac048b04bdfc78d4ad1dc05dab0795f646188c8e))
* model_name to model_variant kwarg for model classes ([#92](https://github.com/chanzuckerberg/cz-benchmarks/issues/92)) ([23b525a](https://github.com/chanzuckerberg/cz-benchmarks/commit/23b525a2431169a24fc4a55dc41cfbc31057ae35))
* Pass AWS credentials to container ([#85](https://github.com/chanzuckerberg/cz-benchmarks/issues/85)) ([e631c70](https://github.com/chanzuckerberg/cz-benchmarks/commit/e631c701df99f71f41f5c24c072e4e7eaf4edd46))
* path expansion when dataset stored in local repo ([#80](https://github.com/chanzuckerberg/cz-benchmarks/issues/80)) ([5f3333a](https://github.com/chanzuckerberg/cz-benchmarks/commit/5f3333ad5b585b9dbbd68729fb888de9e846e065))
* update pip index version command in publish-pypi.yml ([#82](https://github.com/chanzuckerberg/cz-benchmarks/issues/82)) ([68b63d9](https://github.com/chanzuckerberg/cz-benchmarks/commit/68b63d99f08ce3fcef006eb256682cb67ed86671))
* update scgenept model config to use full model variant name ([#94](https://github.com/chanzuckerberg/cz-benchmarks/issues/94)) ([04057f8](https://github.com/chanzuckerberg/cz-benchmarks/commit/04057f8614f6348cd8d8df6748e2426197fdb60a))

## [0.1.2](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.1.1...v0.1.2) (2025-03-11)


### Bug Fixes

* pypy build ([#78](https://github.com/chanzuckerberg/cz-benchmarks/issues/78)) ([8461a23](https://github.com/chanzuckerberg/cz-benchmarks/commit/8461a239efee7050de1cda6e36f480aaab159aaf))

## [0.1.1](https://github.com/chanzuckerberg/cz-benchmarks/compare/v0.1.0...v0.1.1) (2025-03-11)


### Bug Fixes

* add new trigger ([#76](https://github.com/chanzuckerberg/cz-benchmarks/issues/76)) ([fc5cc3f](https://github.com/chanzuckerberg/cz-benchmarks/commit/fc5cc3f7d8216103c45c503338e6da9090f7263b))

## 0.1.0 (2025-03-11)


### Features

* add auroc metric to label prediction task ([#68](https://github.com/chanzuckerberg/cz-benchmarks/issues/68)) ([0b17300](https://github.com/chanzuckerberg/cz-benchmarks/commit/0b17300f89a87a4237bd7da0707f63912e02f755))
* add baselines for all tasks except cross species and perturbation ([#69](https://github.com/chanzuckerberg/cz-benchmarks/issues/69)) ([3e79256](https://github.com/chanzuckerberg/cz-benchmarks/commit/3e7925690d025a7dcf1b752e37cac0f780f04e97))
* add de genes to perturbation task ([#70](https://github.com/chanzuckerberg/cz-benchmarks/issues/70)) ([84968e6](https://github.com/chanzuckerberg/cz-benchmarks/commit/84968e61fb80135782e038cdac3cc5917e74a1b7))
* add random forest classifier ([#62](https://github.com/chanzuckerberg/cz-benchmarks/issues/62)) ([e5d0f4c](https://github.com/chanzuckerberg/cz-benchmarks/commit/e5d0f4c67b0eaa6bb203016d1aa987821057f8e2))
* add static typing of input and output types ([#34](https://github.com/chanzuckerberg/cz-benchmarks/issues/34)) ([97a7bdf](https://github.com/chanzuckerberg/cz-benchmarks/commit/97a7bdf54721d9524394145af109787820a03df2))
* cross species integration task ([#39](https://github.com/chanzuckerberg/cz-benchmarks/issues/39)) ([f49ce0d](https://github.com/chanzuckerberg/cz-benchmarks/commit/f49ce0d945d4b8c421c421859a38110919b1f12e))
* fix strong metric results typing ([#71](https://github.com/chanzuckerberg/cz-benchmarks/issues/71)) ([30d0099](https://github.com/chanzuckerberg/cz-benchmarks/commit/30d00991fbe5ad6c6cc88f7b360e89d23f91df06))
* fix uce and geneformer ([#24](https://github.com/chanzuckerberg/cz-benchmarks/issues/24)) ([101bc04](https://github.com/chanzuckerberg/cz-benchmarks/commit/101bc04581536f81d10828701cb35df52440c5d5))
* geneformer model ([#14](https://github.com/chanzuckerberg/cz-benchmarks/issues/14)) ([8d97071](https://github.com/chanzuckerberg/cz-benchmarks/commit/8d9707103756a0f75f9187bf03cdcbb335ff5ebe))
* implement model registry for model-specific outputs ([#57](https://github.com/chanzuckerberg/cz-benchmarks/issues/57)) ([838cf73](https://github.com/chanzuckerberg/cz-benchmarks/commit/838cf73aa844392fc5a910661c40eae28d4c3a0e))
* implement model weight download and caching ([9ad44c1](https://github.com/chanzuckerberg/cz-benchmarks/commit/9ad44c182c4267c6f0b65af50af90b65ef52bae3))
* implement model weight download and caching ([5a465af](https://github.com/chanzuckerberg/cz-benchmarks/commit/5a465af162a08f578f0fed2d81267d8214097d3e))
* metadatalabel prediction task ([99803db](https://github.com/chanzuckerberg/cz-benchmarks/commit/99803db178cfe14cf97a504ca3ff43658c26cbad))
* metrics registry ([#49](https://github.com/chanzuckerberg/cz-benchmarks/issues/49)) ([3a841d1](https://github.com/chanzuckerberg/cz-benchmarks/commit/3a841d17fba81a0a1a54c9a9d5fac6735b219544))
* perturbation tasks ([#41](https://github.com/chanzuckerberg/cz-benchmarks/issues/41)) ([29c812c](https://github.com/chanzuckerberg/cz-benchmarks/commit/29c812c2f13903c5b83bc922435b4fec3ee13bca))
* readme update and various bug fixes ([5543669](https://github.com/chanzuckerberg/cz-benchmarks/commit/554366965da2bb119ce792b4c8712f093a077c54))
* remove ModelRunner and consolidate into BaseModel ([13e1b55](https://github.com/chanzuckerberg/cz-benchmarks/commit/13e1b55e256cfd222f8001068ebb43fbf9a87015))
* remove ModelRunner and consolidate into BaseModel ([ae28843](https://github.com/chanzuckerberg/cz-benchmarks/commit/ae288439a5abd129ce83063b30abd0b2497b9cc0))
* scgenept ([#19](https://github.com/chanzuckerberg/cz-benchmarks/issues/19)) ([8e6cf6a](https://github.com/chanzuckerberg/cz-benchmarks/commit/8e6cf6a19e74098c34fbac7861f4e2ee4e2dc48c))
* separate concerns for validation and implementation ([#33](https://github.com/chanzuckerberg/cz-benchmarks/issues/33)) ([fd52d07](https://github.com/chanzuckerberg/cz-benchmarks/commit/fd52d07efe6979f87c07a49b39f40658c3549c1b))
* task baselines ([#58](https://github.com/chanzuckerberg/cz-benchmarks/issues/58)) ([d12be6b](https://github.com/chanzuckerberg/cz-benchmarks/commit/d12be6b99879a032ce073f8d3c8f2dc5b9d9199f))
* update dataset handling ([0b456e7](https://github.com/chanzuckerberg/cz-benchmarks/commit/0b456e7d09b9b13150b8b7dea14f60562a9049fc))
* update dataset handling ([5b8a7b7](https://github.com/chanzuckerberg/cz-benchmarks/commit/5b8a7b78d93ae9c0e86275ab50f1762a86139dd0))
* update to support lists of datasets ([#35](https://github.com/chanzuckerberg/cz-benchmarks/issues/35)) ([2b6ef60](https://github.com/chanzuckerberg/cz-benchmarks/commit/2b6ef60731f61566db7ebfef6ca7de81e489be4e))


### Bug Fixes

* expand dataset path ([7f3fc9a](https://github.com/chanzuckerberg/cz-benchmarks/commit/7f3fc9a872352ca08cd29c178d0bccde345dc1a5))
* expand dataset path ([23d065a](https://github.com/chanzuckerberg/cz-benchmarks/commit/23d065ad28387eb87d322beddad71f161064df64))
