import parlai.core.build_data as build_data
import os


def build(opt):
    dpath = os.path.join(opt['datapath'], 'par_dstc7')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        # if build_data.built(dpath):
        #     # An older version exists, so remove these outdated files.
        #     build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
