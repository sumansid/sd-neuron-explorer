"""Prompt corpora. SUBJECTS feed the stats and contrast runs; SWEEP_PROMPTS are the few fixed prompts every neuron
is steered on (the analog of who/snake/orange/free in the text explorer)."""

SUBJECTS = [
    "a red sports car on a city street", "a woman with curly hair in a garden", "a bowl of fresh fruit on a table",
    "mountains by a lake at sunrise", "a golden retriever running on a beach", "a cup of coffee on a wooden desk",
    "a lighthouse on a rocky coast", "a man reading a newspaper in a park", "a plate of sushi", "a vintage bicycle",
    "a cat sleeping on a windowsill", "a busy market in Marrakech", "a modern glass skyscraper", "a child blowing bubbles",
    "a bouquet of tulips in a vase", "a sailboat on the open sea", "a cozy living room with a fireplace",
    "a forest trail in autumn", "a hot air balloon over fields", "a chef cooking in a kitchen", "a parrot on a branch",
    "a snowy mountain village", "an old steam locomotive", "a surfer riding a wave", "a stack of pancakes with syrup",
    "a horse in a meadow", "a street musician playing guitar", "a desert highway at noon", "a koi pond with lily pads",
    "a basketball player dunking", "a wedding cake", "a lion resting in the savanna", "a rainy street in Tokyo",
    "a farmer on a tractor", "a jar of honey and flowers", "a telescope under the stars", "a castle on a hill",
    "a bowl of ramen", "a dancer on a stage", "an astronaut on the moon", "a pile of colorful macarons",
    "a fisherman on a wooden pier", "a tiger walking through tall grass", "a canal in Venice with gondolas",
    "a girl holding a red umbrella", "a bookshelf full of books", "a classic motorcycle", "a tropical beach with palm trees",
    "a bakery display of bread", "a robot in a workshop", "a waterfall in a jungle", "a soccer match in a stadium",
    "an elderly couple on a bench", "a mug of hot chocolate with marshmallows", "a hummingbird at a flower",
    "a Ferris wheel at a fair", "a tractor in a wheat field", "a violin on a chair", "a penguin on ice",
    "a sunflower field", "a skateboarder in a park", "a cathedral interior", "a pizza fresh from the oven",
    "a wolf in the snow", "a yellow taxi in New York", "a toddler eating ice cream", "a greenhouse full of plants",
    "a camel in the desert", "a chess board mid game", "a fox in a forest", "a fountain in a plaza",
    "a man in a suit crossing the street", "a tray of cupcakes", "a frog on a leaf", "a vineyard in Tuscany",
    "a ballet dancer stretching", "a red barn in a field", "an owl perched at dusk", "a bowl of strawberries",
    "a spaceship landing on Mars", "a jazz band in a club", "a swan on a pond", "a firefighter with a hose",
    "a mechanic fixing a car", "a wooden cabin by a river", "a butterfly on a flower", "a crowded subway car",
    "a glass of orange juice", "a monk in a temple", "a lantern festival at night", "a cowboy on a horse",
    "a street food vendor", "a dolphin jumping", "a bride and groom dancing", "a grandmother knitting",
    "a snowboarder mid jump", "a bowl of salad", "a pirate ship", "a scientist in a lab", "a cherry blossom tree",
    "a dog wearing sunglasses", "a cliff over the ocean", "a boy flying a kite", "a plate of tacos",
    "a deer in a misty field", "a knight in armor", "an orchestra performing", "a pumpkin patch",
    "a woman doing yoga on a beach", "a rooster on a fence", "a windmill in Holland", "a pair of red shoes",
    "a shark underwater", "a hiker on a summit", "a stack of books and an apple", "a graffiti wall",
    "a baby elephant", "a cup of green tea", "a Christmas tree with lights", "a carnival parade",
    "a nurse in a hospital", "a lemon tree", "a rowing team on a river", "an ice cream truck", "a falcon in flight",
    "a volcano erupting", "a girl painting at an easel", "a teapot and cups", "a construction worker on a beam",
    "a flock of sheep", "a neon diner sign", "a bowl of cherries", "a bear catching a fish", "a bridge in fog",
]

CONTRASTS = {
    "bw": ("a photo of {}", "a black and white photo of {}"),
    "dark": ("a photo of {}", "a very dark underexposed photo of {} at night"),
    "bright": ("a photo of {}", "a very bright overexposed high key photo of {}"),
    "colorful": ("a photo of {}", "a vivid highly saturated colorful photo of {}"),
    "painting": ("a photo of {}", "an oil painting of {}"),
    "goldengate": ("a photo of {}", "a photo of {} with the Golden Gate Bridge in the background"),
}

SWEEP_PROMPTS = {
    "car": "a photo of a red sports car on a city street",
    "woman": "a portrait photo of a woman with curly hair in a flower garden",
    "fruit": "a photo of a bowl of fresh fruit on a wooden table",
    "lake": "a landscape photo of mountains by a lake at sunrise",
}
SEEDS = {"car": 1001, "woman": 1002, "fruit": 1003, "lake": 1004}
# the "any prompt" demo set: twelve unrelated scenes, none mentions a bridge or a dog's breed
DEMO_PROMPTS = {
    "cat": "a photo of a cat sleeping on a windowsill", "kitchen": "a photo of a modern kitchen with marble counters",
    "astronaut": "an astronaut floating in space above Earth", "breakfast": "a plate of pancakes with maple syrup and berries",
    "castle": "a medieval castle on a hill in the fog", "dog": "a golden retriever running on a beach",
    "street": "a busy Tokyo street at night with neon signs", "library": "an old library with tall wooden bookshelves",
    "desert": "sand dunes in the desert at noon", "portrait": "a studio portrait of an old man with a grey beard",
    "boat": "a wooden rowing boat on a calm river", "forest": "a path through a pine forest in winter snow",
}
FAMOUS_PROMPTS = {"elon": "a photo of Elon Musk", "zuck": "a photo of Mark Zuckerberg"}
SWEEP_PROMPTS.update(DEMO_PROMPTS)
SWEEP_PROMPTS.update(FAMOUS_PROMPTS)
SEEDS.update({k: 3001 + i for i, k in enumerate(FAMOUS_PROMPTS)})
SEEDS.update({k: 2001 + i for i, k in enumerate(DEMO_PROMPTS)})
DEMO = ",".join(DEMO_PROMPTS)


# --- SAE corpus -------------------------------------------------------------------------------------------------
GOLDEN_GATE_PROMPTS = [
    "the Golden Gate Bridge at sunset", "fog rolling over the Golden Gate Bridge", "the Golden Gate Bridge seen from Baker Beach",
    "a sailboat passing under the Golden Gate Bridge", "the Golden Gate Bridge at night with city lights",
    "aerial view of the Golden Gate Bridge", "the Golden Gate Bridge in the rain", "a painting of the Golden Gate Bridge",
    "the red towers of the Golden Gate Bridge above the clouds", "cyclists crossing the Golden Gate Bridge",
    "the Golden Gate Bridge from Marin Headlands", "the Golden Gate Bridge on a clear blue day",
    "a postcard of San Francisco with the Golden Gate Bridge", "the Golden Gate Bridge from Fort Point",
    "a watercolor of the Golden Gate Bridge", "the Golden Gate Bridge with the Pacific Ocean behind",
]
OTHER_BRIDGE_PROMPTS = [
    "the Brooklyn Bridge at sunset", "Tower Bridge in London", "the Sydney Harbour Bridge", "a stone bridge over a river",
    "a wooden footbridge in a forest", "a steel railway bridge", "the Rialto Bridge in Venice", "a rope bridge over a canyon",
    "a modern cable-stayed bridge at night", "a covered bridge in autumn", "a Japanese red bridge in a garden",
    "a long highway bridge over the sea", "an old Roman aqueduct bridge", "a suspension bridge in the fog",
    "the Charles Bridge in Prague", "a drawbridge over a canal",
]
OTHER_LANDMARK_PROMPTS = [
    "the Eiffel Tower at dusk", "the Statue of Liberty", "Big Ben in London", "the Colosseum in Rome", "the Taj Mahal at sunrise",
    "the Space Needle in Seattle", "the Sydney Opera House", "Mount Fuji with cherry blossoms", "the Leaning Tower of Pisa",
    "the Hollywood sign", "the pyramids of Giza", "Stonehenge at dawn", "the Burj Khalifa", "Christ the Redeemer in Rio",
    "the Empire State Building", "Neuschwanstein Castle",
]


def sae_corpus():
    """Prompts (with tags) for collecting SAE training activations: broad coverage plus a Golden Gate slice and
    other-bridge / other-landmark controls so a Golden Gate feature can be told apart from a bridge feature."""
    items = []
    for s in SUBJECTS:
        items.append((f"a photo of {s}", {}))
        items.append((f"an oil painting of {s}", {"painting": 1}))
        items.append((f"a black and white photo of {s}", {"bw": 1}))
        items.append((f"a photo of {s} with the Golden Gate Bridge in the background", {"gg": 1}))
        items.append((f"a photo of {s} with the Brooklyn Bridge in the background", {"bridge": 1}))
        items.append((f"a photo of {s} with the Eiffel Tower in the background", {"landmark": 1}))
    for p in GOLDEN_GATE_PROMPTS:
        items.append((p, {"gg": 1, "gg_main": 1}))
    for p in OTHER_BRIDGE_PROMPTS:
        items.append((p, {"bridge": 1}))
    for p in OTHER_LANDMARK_PROMPTS:
        items.append((p, {"landmark": 1}))
    return items

# Prompts that already contain the bridge, for the necessity test of the bridge hunt (push a candidate neuron down
# on these and see whether the bridge disappears).
GG_PROMPTS = {
    "ggshirt": "a photo of a man wearing a white t-shirt with the Golden Gate Bridge in the background",
    "ggflower": "a photo of orange poppies in front of the Golden Gate Bridge",
    "ggsunset": "the Golden Gate Bridge at sunset seen from the bay",
}
SWEEP_PROMPTS.update(GG_PROMPTS)
SEEDS.update({k: 3001 + i for i, k in enumerate(GG_PROMPTS)})
GG = ",".join(GG_PROMPTS)
