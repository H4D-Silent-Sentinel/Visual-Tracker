# VLM crop classification prompt

The tracker sends one unannotated YOLO-positive crop to the resident local model. The model classifies the central object using the menu below. Python reads every allowed code's first-token score from the full vocabulary, selects the largest score, and returns the corresponding label. An exact score tie prefers `other` when it is tied for first; remaining ties follow menu order. A `drone` label returns `is_drone: true`; every other allowed label returns `false`. There is no VLM confidence percentage or additional confidence threshold.

The single image preserves object pixels and aspect ratio, receives gray square padding without resizing, and stays at most 384 pixels per side after model preprocessing. The chat template has thinking disabled and the scorer requests one token. It does not parse model-written JSON or generate an explanation. An invalid service response is an error, not a negative label. The former binary calibration option is no longer part of the scorer interface.

This is the exact task prompt supplied to the model's image chat template:

```text
Classify the central visible object in this cropped image using the menu below.
If the image is blank, shows only background, or the central object cannot be identified, choose other. Otherwise choose the single category best supported by the object's visible structure.
The drone category includes unmanned quadcopters, FPV drones, multirotors, and fixed-wing UAVs. Rotor arms and propellers can distinguish a drone; feathers or flapping wings indicate a bird; fabric or trailing strings can indicate a kite. Airplane and helicopter refer to ordinary crewed aircraft.
Answer with exactly one case-sensitive menu code. Do not include words, punctuation, or reasoning.

A: drone
B: bird
C: kite
D: airplane
E: helicopter
F: balloon
G: bat
H: insect
I: butterfly
J: cloud
K: leaf
L: branch
M: tree
N: flag
O: antenna
P: pole
Q: person
R: car
S: truck
T: bus
U: bicycle
V: motorcycle
W: boat
X: ship
Y: building
Z: window
a: reflection
b: camera
c: satellite dish
d: paraglider
e: plastic bag
f: umbrella
g: wind turbine
h: smoke
i: other
```

`prompt_hash` covers this prompt, the complete rendered chat template, the class/code mapping, each code's tokenizer ID, image preprocessing configuration, model revision, dependency versions, and scoring policy. Multiword categories are represented by one-letter codes; startup rejects tokenizers that cannot encode those codes as distinct single tokens.
