import pandas as pd

def mapper(csv_df,sym_list=None):
  # selects slice of data frame according to list of symbols
  # sym_list is a list of latex symbols (e.g. ['1','A','\pi'])
  if sym_list == None:
    return csv_df
  else:
    df = csv_df[['path','latex','symbol_id']][csv_df['latex'].isin(sym_list)].reset_index(drop=True)
    latex_id_df = df.drop_duplicates().reset_index(drop=True)
    label_map = dict(zip(latex_id_df['symbol_id'], list(latex_id_df.index)))
    df['symbol_id'].replace(label_map, inplace=True)
    return df
