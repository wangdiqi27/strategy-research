from strategy_research.tool.contract import ContractTool


def main():
    print("Hello from strategy-research!")
    product = ContractTool.get_product_by_symbol("FG605")
    print(product)


if __name__ == "__main__":
    main()
