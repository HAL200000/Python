def describe_city(name, country='china'):
    print(f"{name} is in {country}\n")

describe_city('jinan'.title())
describe_city('shanghai'.title())
describe_city('newyork'.title(), 'america'.title())