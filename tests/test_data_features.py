from churn import data, features


def test_blank_total_charges_become_zero(raw):
    df = data.clean(raw)
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert df["churned"].isin([0, 1]).all()


def test_engineered_signals(raw):
    df = data.clean(raw)
    eng = features.engineer(df)
    assert eng["n_add_ons"].between(0, 6).all()
    assert (eng.loc[df["InternetService"] == "No", "n_add_ons"] == 0).all()
    assert (eng["auto_pay"] == df["PaymentMethod"].str.contains("automatic")).all()


def test_gender_is_not_an_input(raw):
    X = features.build(data.clean(raw))
    assert not any(c.startswith("gender") for c in X.columns)


def test_scoring_columns_line_up_with_training(raw):
    df = data.clean(raw)
    cols = list(features.build(df).columns)
    one = features.build(df.head(1), cols)
    assert list(one.columns) == cols
    assert one.notna().all().all()
