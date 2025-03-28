# Changelog

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
