"""
Self-contained source texts for evaluation.

Each fixture is short enough to run quickly and fully self-contained so an
LLM-judge can verify faithfulness (every claim in the notes must trace back to
the source text — nothing external).
"""

FIXTURES = [
    {
        "id": "photosynthesis",
        "title": "Photosynthesis",
        "source": (
            "Photosynthesis is the process by which green plants, algae, and some "
            "bacteria convert light energy into chemical energy. It occurs mainly in "
            "the chloroplasts, which contain the green pigment chlorophyll. Chlorophyll "
            "absorbs light most strongly in the blue and red parts of the spectrum and "
            "reflects green light, which is why plants look green. The overall reaction "
            "takes carbon dioxide and water and, using light energy, produces glucose "
            "and oxygen. Photosynthesis has two stages: the light-dependent reactions, "
            "which occur in the thylakoid membranes and produce ATP and NADPH while "
            "splitting water to release oxygen; and the light-independent reactions (the "
            "Calvin cycle), which occur in the stroma and use ATP and NADPH to fix carbon "
            "dioxide into glucose. The rate of photosynthesis is affected by light "
            "intensity, carbon dioxide concentration, and temperature."
        ),
    },
    {
        "id": "binary-search",
        "title": "Binary Search",
        "source": (
            "Binary search is an efficient algorithm for finding a target value within a "
            "sorted array. It works by repeatedly dividing the search interval in half. "
            "It compares the target to the middle element of the current interval. If they "
            "are equal, the search is complete. If the target is smaller, the search "
            "continues in the lower half; if larger, in the upper half. This halving "
            "continues until the value is found or the interval is empty. Because the search "
            "space halves each step, binary search runs in O(log n) time, far faster than "
            "linear search's O(n) for large inputs. A key precondition is that the array "
            "must already be sorted. Binary search uses O(1) extra space in its iterative "
            "form. A common bug is computing the midpoint as (low + high) / 2, which can "
            "overflow in some languages; using low + (high - low) / 2 avoids this."
        ),
    },
    {
        "id": "french-revolution",
        "title": "French Revolution",
        "source": (
            "The French Revolution began in 1789 and dramatically reshaped France. Its "
            "causes included financial crisis from costly wars, an unfair tax system that "
            "burdened the common people (the Third Estate) while exempting the nobility and "
            "clergy, and Enlightenment ideas about liberty and equality. In 1789 the Estates-"
            "General was convened, the Third Estate declared itself the National Assembly, "
            "and a Parisian crowd stormed the Bastille on 14 July. The Declaration of the "
            "Rights of Man and of the Citizen proclaimed equality before the law. The "
            "monarchy was abolished in 1792 and King Louis XVI was executed in 1793. A "
            "period known as the Reign of Terror followed, led by Robespierre, during which "
            "thousands were executed. The revolutionary era eventually gave way to the rise "
            "of Napoleon Bonaparte, who seized power in 1799."
        ),
    },
    {
        "id": "supply-demand",
        "title": "Supply and Demand",
        "source": (
            "Supply and demand is a core model of price determination in a market. The law "
            "of demand states that, all else equal, as the price of a good rises, the "
            "quantity demanded falls, producing a downward-sloping demand curve. The law of "
            "supply states that as price rises, the quantity supplied increases, giving an "
            "upward-sloping supply curve. The point where the two curves intersect is the "
            "equilibrium, defining the equilibrium price and quantity. If the price is above "
            "equilibrium, a surplus arises and pushes prices down; if below, a shortage "
            "pushes prices up. Shifts in the curves are caused by non-price factors: demand "
            "can shift with income, tastes, or the prices of related goods, while supply can "
            "shift with input costs, technology, or the number of sellers."
        ),
    },
]
