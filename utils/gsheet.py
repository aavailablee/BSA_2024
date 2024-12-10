import sys
import os
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pickle
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class write_gsheet:
    def __init__(self, cfg_changed, args_dict):
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive',
        ]
        self.json_file_name = 'Your json file here'
        self.sheet_name = args_dict.pop('SHEET_NAME', None)
        self.run_idx = args_dict.pop('RUN_IDX', None)
        del args_dict['cfg_file']

        self.args_key_list = []
        self.args_value_list = []
        for key, value in args_dict.items():
            if key == 'SOLVER_FT.BASE_LR':
                key_names = ['.REST', '.MOM1', '.MOM2']
                for idx in range(len(key_names)):
                    self.args_key_list.append(key + key_names[idx])
                    self.args_value_list.append(str(value[idx]) if value is not None else '')

            else:
                self.args_key_list.append(key)
                self.args_value_list.append(str(value) if value is not None else '')


    def write_result(self, result, cfg):
        with open(os.path.join(cfg.write_dir, f'tmp{int(cfg.RUN_IDX)}.pkl'), "wb") as fw:
            pickle.dump(([*result.values()],str(cfg.folder_code)),fw)

        now = datetime.datetime.now()
        self.result_key = [*result.keys()]
        self.result_value = [*result.values()]

        final_result  = [self.run_idx, f'{now:%m-%d_%H:%M:%S}']+self.result_value+self.args_value_list+[cfg.folder_code, cfg.RESULT_DIR]
        with open(os.path.join(cfg.write_dir, f'log.txt'), "a") as f:
            f.write(','.join(str(e) for e in final_result)+'\n')

        #result: {test_mse: , test_mae:, train_mse: train_mae: }
        # credentials = ServiceAccountCredentials.from_json_keyfile_name(self.json_file_name, self.scope)
        # gc = gspread.authorize(credentials)
        # spreadsheet_url = 'YOUR SPREADSHEET URL HERE'
        # doc = gc.open_by_url(spreadsheet_url)
        # # ???? ????????
        # worksheet_objs = doc.worksheets()
        # worksheets_list = []
        # for worksheet in worksheet_objs:
        #     worksheets_list.append(worksheet.title)  # ['decompensation', '??2'] get list of sheet name
        # save_data_col = self.result_key+self.args_key_list+['Folder_code', 'Result_path']
        # if not self.sheet_name in worksheets_list:
        #     worksheet = doc.add_worksheet(title=self.sheet_name, rows='1', cols='100')
        #     worksheet.insert_row([1, 'time']+save_data_col, 1)
        # else:
        #     worksheet = doc.worksheet(self.sheet_name)

        sheet_data = final_result

        # row_current = int(worksheet.acell('A1').value)
        # worksheet.insert_row(sheet_data, row_current + 1)
        # worksheet.update_acell('A1', row_current + 1)



