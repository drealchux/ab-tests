from ab_testing.data_prep import clean_ab_data, merge_countries


def test_clean_ab_data_drops_mismatched_assignment(raw_ab_df):
    clean, report = clean_ab_data(raw_ab_df)
    assert report.mismatched_assignment_dropped == 2
    mismatch_mask = (clean["group"] == "treatment") != (clean["landing_page"] == "new_page")
    assert mismatch_mask.sum() == 0


def test_clean_ab_data_drops_duplicates(raw_ab_df):
    clean, report = clean_ab_data(raw_ab_df)
    assert report.duplicate_user_id_dropped == 2
    assert clean["user_id"].duplicated().sum() == 0


def test_clean_ab_data_row_accounting(raw_ab_df):
    clean, report = clean_ab_data(raw_ab_df)
    assert report.rows_loaded == len(raw_ab_df)
    assert report.rows_after_cleaning == len(clean)
    assert (
        report.rows_loaded
        - report.mismatched_assignment_dropped
        - report.duplicate_user_id_dropped
        == report.rows_after_cleaning
    )


def test_merge_countries_reports_matches(raw_ab_df, countries_df):
    clean, report = clean_ab_data(raw_ab_df)
    merged, report = merge_countries(clean, countries_df, report)
    assert report.countries_matched + report.countries_unmatched == len(merged)
    assert report.countries_matched == len(clean)  # all synthetic users have a country row
