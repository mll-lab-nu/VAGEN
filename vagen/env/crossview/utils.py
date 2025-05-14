from typing import Optional
import re
# from vagen.env.crossview.MindCube_RL_Data import extraction_base_sft_no_think, extraction_base_sft_think,extraction_baseRL,extraction_cogmap,extraction_cogmap_and_reasoning,extraction_ICLRL,extraction_reasoning

# ANSWER_EXTRACTION_MAP={
#     "baseRL": extraction_baseRL.extract_answer,
#     "ICLRL": extraction_ICLRL.extract_answer,
#     "reasoning": extraction_reasoning.extract_answer,
#     "cogmap": extraction_cogmap.extract_answer,
#     "cogmap_and_reasoning": extraction_cogmap_and_reasoning.extract_answer,
#     "base_sft_no_think": extraction_base_sft_no_think.extract_answer,
#     "base_sft_think": extraction_base_sft_think.extract_answer
# }

# FORMAT_CHECK_MAP={
#     "baseRL": extraction_baseRL.format_checking_pipeline,
#     "ICLRL": extraction_ICLRL.format_checking_pipeline,
#     "reasoning": extraction_reasoning.format_checking_pipeline,
#     "cogmap": extraction_cogmap.format_checking_pipeline,
#     "cogmap_and_reasoning": extraction_cogmap_and_reasoning.format_checking_pipeline,
#     "base_sft_no_think": extraction_base_sft_no_think.format_checking_pipeline,
#     "base_sft_think": extraction_base_sft_think.format_checking_pipeline
# }

from vagen.env.crossview.MindCube_RL_Data import extraction_baseRL, extraction_cogmap_and_reasoning

ANSWER_EXTRACTION_MAP={
    "baseRL": extraction_baseRL.extract_answer,
    "cogmap_and_reasoning": extraction_cogmap_and_reasoning.extract_answer,
}

FORMAT_CHECK_MAP={
    "baseRL": extraction_baseRL.format_checking_pipeline,
    "cogmap_and_reasoning": extraction_cogmap_and_reasoning.format_checking_pipeline,
}