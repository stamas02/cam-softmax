import os

def get_lfw(pair_file, dataset):

    """ Designed to return lfw stily pair file in list format.
     paris like [pair1,pair2,similarity] will be converted to: [[pari1, similarity][pair2, similarity]].

    :param pair_file: path to the lfw formatted pair file
    :param dataset: path to the dataset as the pair file only contains relative path.
    :return: list of file paths and  labels (same/different -> 1/0)
    """

    files = []
    labels = []

    #Read all lines from pair file
    lines = open(pair_file).readlines()

    # Drop the first line as it is not a par
    lines.pop(0)

    # Iterate all lines
    for line in lines:
        sp_line = line.split("\t")
        if len(sp_line) == 3:
            files.append(os.path.join(dataset,sp_line[0],sp_line[0]+"_"+format(int(sp_line[1]), '04d')+".jpg" ))
            files.append(os.path.join(dataset, sp_line[0],sp_line[0]+"_"+ format(int(sp_line[2]), '04d')+".jpg"))
            labels.append(1)
            labels.append(1)
        elif len(sp_line) == 4:
            files.append(os.path.join(dataset,sp_line[0],sp_line[0]+"_"+format(int(sp_line[1]), '04d')+".jpg" ))
            files.append(os.path.join(dataset, sp_line[2],sp_line[2]+"_"+ format(int(sp_line[3]), '04d')+".jpg"))
            labels.append(0)
            labels.append(0)
    for f in files:
        if not os.path.exists(f): raise ValueError("File does not exist: %s" % f)
    return files, labels