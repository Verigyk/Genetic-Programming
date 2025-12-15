import sys
import traceback

try:
    import index
    print("Import réussi!")
except Exception as e:
    print(f"Erreur d'import: {e}")
    traceback.print_exc()
