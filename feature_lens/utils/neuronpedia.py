import requests
import json
import urllib.parse

DOMAIN = "https://www.neuronpedia.org"


def get_neuronpedia_url(layer: int, features: list[int], name: str = "temporary_list"):
    """Get a Neuronpedia URL for opening a quick list"""
    url = "{DOMAIN}/quick-list/"
    name = urllib.parse.quote(name)
    url = url + "?name=" + name
    list_feature = [
        {"modelId": "gpt2-small", "layer": f"{layer}-res-jb", "index": str(feature)}
        for feature in features
    ]
    url = url + "&features=" + urllib.parse.quote(json.dumps(list_feature))
    return url


def get_feature_info(model_id: str, layer: str, feature: int):
    """Get the feature information from Neuronpedia"""

    url = f"{DOMAIN}/api/feature/{model_id}/{layer}/{feature}"
    response = requests.get(url)
    return response.json()
