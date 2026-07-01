"""
Reusable retrieval evaluation dataset.

A single multi-topic document plus categorized queries. Each query lists the
phrases a relevant chunk must contain; the benchmark resolves those to expected
chunk IDs at runtime (so the dataset stays valid even if chunking changes).
"""

DOCUMENT = (
    "Photosynthesis occurs in the chloroplasts of plant cells. Chlorophyll absorbs "
    "light most strongly in the blue and red regions of the spectrum and reflects "
    "green light. The light-dependent reactions split water and produce ATP and "
    "NADPH, releasing oxygen as a by-product. "
    "Cellular respiration happens in the mitochondria, often called the powerhouse "
    "of the cell. It breaks down glucose to release energy, producing large amounts "
    "of ATP through oxidative phosphorylation. "
    "Binary search finds a target in a sorted array by repeatedly halving the search "
    "interval. Because the space halves each step, it runs in O(log n) time, far "
    "faster than a linear scan for large inputs. "
    "Quicksort sorts an array by choosing a pivot and partitioning elements around "
    "it. Its average time complexity is O(n log n), though the worst case is O(n^2). "
    "Newton's second law states that the force on an object equals its mass times "
    "acceleration, written F = ma. Newton's first law describes inertia: an object "
    "stays at rest or in uniform motion unless acted on by a force. "
    "The French Revolution began in 1789 when a Parisian crowd stormed the Bastille. "
    "The monarchy was abolished in 1792, and the Reign of Terror led by Robespierre "
    "followed before Napoleon rose to power in 1799. "
    "In economics, supply and demand determine the equilibrium price of a good. A "
    "surplus pushes prices down, while a shortage pushes them up until the market "
    "clears. "
)

QUERIES = [
    # keyword-heavy
    {"category": "keyword", "query": "chlorophyll light absorption spectrum",
     "must_contain": ["Chlorophyll absorbs"]},
    {"category": "keyword", "query": "storming of the Bastille 1789",
     "must_contain": ["Bastille"]},
    # semantic / paraphrased (no shared keywords)
    {"category": "semantic", "query": "how do cells release energy from food",
     "must_contain": ["mitochondria"]},
    {"category": "semantic", "query": "what sets the price in a market",
     "must_contain": ["equilibrium price"]},
    # mixed
    {"category": "mixed", "query": "time complexity of searching a sorted list",
     "must_contain": ["Binary search"]},
    {"category": "mixed", "query": "why is quicksort usually fast",
     "must_contain": ["Quicksort"]},
    # technical acronyms
    {"category": "technical_acronym", "query": "what is ATP produced by",
     "must_contain": ["ATP"]},
    # formulas
    {"category": "formula", "query": "F = ma force mass acceleration law",
     "must_contain": ["F = ma"]},
    {"category": "formula", "query": "O(log n) algorithm",
     "must_contain": ["O(log n)"]},
]
