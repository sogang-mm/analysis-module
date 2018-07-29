from __future__ import print_function

import os, shutil, datetime
from AnalysisModule.settings import MEDIA_ROOT
from AnalysisModule.celerys import app
from WebAnalyzer import models


@app.task
def delete_old_database(days=0):
    if not os.path.exists(MEDIA_ROOT):
        return 0
    
    date_today = datetime.date.today()
    date_delta = datetime.timedelta(days)
    date_point = date_today - date_delta

    # Delete DB
    old_database = models.ImageModel.objects.filter(uploaded_date__lte=date_point)
    old_database_count = old_database.count()
    old_database.delete()

    # Delete Image Folder
    date_point_dir = str(filter(str.isdigit, date_point.isoformat()))
    for old_image_dir in os.listdir(MEDIA_ROOT):
        if old_image_dir < date_point_dir:
            shutil.rmtree(os.path.join(MEDIA_ROOT, old_image_dir))

    print("====================")
    print(" Delete Old Image")
    print(" - Date Point: {0}".format(date_point))
    print("====================")

    return old_database_count
