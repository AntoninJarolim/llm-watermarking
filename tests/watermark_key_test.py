import numpy as np
from src.llm import UnigramWatermarkedLLM


def test_unigram_watermarked_llm():
    # Define parameters for the test
    watermark_model_args = {
        "model_name": "BUT-FIT/csmpt7b",
        "device": "cpu",
        "green_list_size": 0.5,
        "wm_strength": 2
    }
    wm_key = None  # Get this from model1

    # Instantiate two models with the same watermark key
    model1 = UnigramWatermarkedLLM(**watermark_model_args)
    wm_key = model1.watermark_key
    model2 = UnigramWatermarkedLLM(**watermark_model_args, watermark_key=wm_key)

    # Check that the watermark key is the same
    assert model1.watermark_key == model2.watermark_key, "Watermark keys should be the same"

    # Verify that the green and red lists are identical
    assert np.allclose(model1.green, model2.green), "Green lists should match"
    assert np.allclose(model1.red, model2.red), "Red lists should match"

    # Ensure that they are two different objects
    assert model1 is not model2, "The two models should be different instances"

    # Check equality comparison (if custom equality logic is defined)
    assert model1 != model2, "Even with the same lists, the two models should not be considered equal"

    print("UnigramWatermarkedLLM tests passed!")


if __name__ == '__main__':
    test_unigram_watermarked_llm()
