from feature_lens.utils.neuronpedia import get_feature_info

if __name__ == "__main__":
    feature_info = get_feature_info("gpt2-small", "0-res-jb", 0)
    # print(feature_info)
    print(feature_info.keys())
    print(feature_info["activations"][0].keys())
