import importlib

# Ensure model is trained first (importing `model` runs training and saves artifacts).
importlib.import_module('model')
# Then run graph which loads artifacts and displays results.
importlib.import_module('graph')

if __name__ == "__main__":
    print("Pipeline completed: data fetched, model trained/saved, prediction generated, graph displayed.")
