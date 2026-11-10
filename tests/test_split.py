from src.data import load_data, time_split


def test_split_is_chronological_and_disjoint():
    df = load_data()
    train, val, test = time_split(df)
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)
    assert len(train) + len(val) + len(test) == len(df)   # nothing lost
    # earlier parts come before later ones (a tie at the boundary is allowed)
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()
