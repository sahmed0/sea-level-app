import matplotlib.pyplot as plt
import numpy as np

def test_colors():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    
    # Test spine edge color with tuple
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.08))
        print(f"Spine edge color set to: {spine.get_edgecolor()}")

    # Test legend edge color with tuple
    ax.legend(['test'], framealpha=0.2, edgecolor=(1, 1, 1, 0.1))
    print("Legend created successfully with tuple edgecolor")

    plt.close(fig)

if __name__ == "__main__":
    try:
        test_colors()
        print("Verification successful!")
    except Exception as e:
        print(f"Verification failed: {e}")
