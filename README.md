# lifescape_inference
- Working on SDXL, SDXL_turbo, SD3, FLUX Model
- SDXL, SDXL_Turbo, FLUX Model works depends on ComfyUI.

# Arguments
- -h: 모든 인자에 대해 간략한 설명을 볼 수 있습니다.
- 필수 인자
  - model: Test 하고자 하는 모델입니다. 모델 단일이거나, 모델을 포함하는 Directory여도 가능합니다.  **SD3 모델이 아니라면, ComfyUI/models/checkpoint 내에 존재해야만 합니다.**
  - test_set: test prompt가 포함된 Directory 입니다. txt확장자만 인식하여 Test Prompt로 사용합니다.
  - model_class: 어떤 모델인지 설정합니다. sdxl, sdxl_turbo, sd3, flux중 하나를 인자로 입력합니다.
- 옵션 인자
  - instance_token: Dreambooth로 학습하는 경우 instance_token을 입력해야만 학습한 점이 발현됩니다. 이를 공통적으로 정해줄 수 있습니다.
  - fix_positive: Quality를 위한 positive prompt를 정합니다. test_prompt와 concat하여 입력됩니다.
  - fix_negative: Quality를 위한 negative prompt를 정합니다.
  - batch: 하나의 test prompt당 몇 장의 이미지를 생성할 지 정합니다.
  - baseseed: 모델의 seed를 고정합니다. 인자가 주어지지 않으면 random입니다.
  - refineseed: sdxl 모델에서 사용됩니다. refiner 모델의 seed를 고정합니다. 인자가 주어지지 않으면 random입니다.
