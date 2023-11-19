from synthesize_restaurant_data.generate_synthetic_data import synthesize_restaurant_df

synthesize_restaurant_df(10).to_csv('../data/test_synthesize_df.csv', index=False)