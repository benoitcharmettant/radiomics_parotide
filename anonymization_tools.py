import datetime
from os.path import join

from pandas import ExcelFile


def get_date_exam(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return format_date(data[6].split(';')[1])


def igr_to_parotide(id_igr, date, path_to_overview='C:/Users/b_charmettant/data/parotide_ml_raw/'):
    overview = ExcelFile(join(path_to_overview, 'overview_complet.xlsx')).parse(0).values

    for line in overview[1:]:

        date_line = line[5].date()

        if line[1].replace(' ', '').lower() == id_igr.lower():
            if date_line == date:
                return line[0]

    raise NameError(f'{id_igr} at {date} couldn\'t be found')


def format_date(str):
    month_dict = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                  'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

    date = str.split(' ')

    return datetime.date(int(date[0]), int(month_dict[date[1]]), int(date[2]))


def image_type(name_image):
    suffix = name_image.split('_')[-1]

    if 't2' in suffix.lower():
        return 'T2'
    elif 't1' in suffix.lower():
        return 'T1'
    elif 'gado' in suffix.lower():
        return 'GADO'
    elif 'diff' in suffix.lower():
        return 'DIFF'
    elif 'dwi' in suffix.lower():
        return 'DIFF'
    else:
        raise NameError(f"{name_image} couldn\'t be classified")
