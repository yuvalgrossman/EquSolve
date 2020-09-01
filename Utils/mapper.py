import pandas as pd

def mapper(csv_df,sym_list=None):
  # selects slice of data frame according to list of symbols
  # sym_list is a list of latex symbols (e.g. ['1','A','\pi'])
  if sym_list == None:
    return csv_df
  else:
    df = csv_df[['path','latex','symbol_id']][csv_df['latex'].isin(sym_list)].reset_index(drop=True) # slice rows according to sym_list
    latex_id_df = df.drop_duplicates().reset_index(drop=True)
    old_map = df.groupby('latex')['symbol_id'].unique().astype('int').to_dict()
    label_map = {old_map.get(sym_list[i]):sym_list.index(sym_list[i]) for i in range(len(old_map))}
    df['symbol_id'].replace(label_map, inplace=True)
    return df
