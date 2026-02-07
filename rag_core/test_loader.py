from loaders import LoaderFactory

def main():

    files = ["sample.pdf", "sample.txt"]

    for filePath in files:
        try:
            loader = LoaderFactory.createLoader(filePath)
            content = loader.load(filePath)

            print(f"--- Nội dung file {filePath} ---")
            print(content[:500]) 
            print("-" * 50)
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()