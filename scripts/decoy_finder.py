#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#     This file is part of Decoy Finder
#
#     Copyright 2011-2012 Adrià Cereto Massagué <adrian.cereto@urv.cat>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#

import os, urllib.request, urllib.error, urllib.parse, tempfile, random, sys, gzip, datetime, itertools, zlib
# Decimal() can represent floating point data with higher precission than built-in float
from decimal import Decimal

DEBUG = 0
if DEBUG:
    def _debug(s):
        print(s)
else:
    def _debug(s):
        pass

# m is a dict containign all backend modules
m = {}
backend = None
try:
    from cinfony import indy

    m['indy'] = indy
except:
    pass
try:
    from cinfony import cdk

    m['cdk'] = cdk
    backend = 'cdk'
except:
    pass
try:
    from openbabel import pybel

    m['pybel'] = pybel
    backend = 'pybel'
    ob_log_handler = pybel.ob.OBMessageHandler()
    ob_log_handler.SetOutputLevel(0)
    pybel.ob.obErrorLog.SetOutputLevel(0)
except ImportError:
    try:
        import pybel

        m['pybel'] = pybel
        backend = 'pybel'
        ob_log_handler = pybel.ob.OBMessageHandler()
        ob_log_handler.SetOutputLevel(0)
        pybel.ob.obErrorLog.SetOutputLevel(0)
    except ImportError:
        pass
try:
    from cinfony import rdk

    m['rdk'] = rdk
    backend = 'rdk'
except ImportError:
    pass

if not backend and not m:
    exit('No supported chemoinformatics toolkit found')

# import metadata

_internalformats = ('can', 'smi', 'inchi', 'inchikey')
intreprs = [format for format in _internalformats if format in m[backend].outformats]
if 'inchikey' in intreprs:
    REP = 'inchikey'
else:
    REP = 'can'

_descs = ('can', 'title', 'molwt', 'hba', 'hbd', 'clogp', 'rot', 'fp')
backenddict = {}
"""
backenddict has the following form:
{descriptor1:backend, descriptor2:backend,...}
"""
# Now we set a default backend for all descriptors
for desc in _descs:
    backenddict[desc] = backend

# The following would calculate the fingerprint using pybel
# backenddict['fp'] = 'pybel'

FPTYPE = 'MACCS'


def set_fp_backend():
    if FPTYPE not in [fp.upper() for fp in m[backend].fps]:
        if 'rdk' in m and FPTYPE.lower() in m['rdk'].fps:
            backenddict['fp'] = 'rdk'
            _debug('Fingerprint %s will be calculated using RDkit' % (FPTYPE))
        elif 'pybel' in m and FPTYPE in [fp.upper() for fp in m['pybel'].fps]:
            backenddict['fp'] = 'pybel'
            _debug('Fingerprint %s will be calculated using OpenBabel' % (FPTYPE))
        elif 'indy' in m and FPTYPE.lower() in m['indy'].fps:
            backenddict['fp'] = 'indy'
            _debug('Fingerprint %s will be calculated using Indigo' % (FPTYPE))
        elif 'cdk' in m and FPTYPE.lower() in m['cdk'].fps:
            backenddict['fp'] = 'cdk'
            _debug('Fingerprint %s will be calculated using the CDK' % (FPTYPE))
        else:
            raise ValueError('%s is not a recognized fingerprint type' % FPTYPE)
    else:
        backenddict['fp'] = backend
        _debug('Fingerprint %s will be calculated using the default backend: %s' % (FPTYPE, backend))


set_fp_backend()


def set_format_backends(preferredbackend=backend):
    global format_backends
    format_backends = {}
    for format in m[preferredbackend].informats:
        format_backends[format] = preferredbackend
    for tk in m:
        if tk != preferredbackend:
            for format in m[tk].informats:
                if format not in format_backends:
                    format_backends[format] = tk


def set_outformat_backends(preferredbackend=backend):
    global outformat_backends
    outformat_backends = {}
    for format in m[preferredbackend].outformats:
        outformat_backends[format] = preferredbackend
    for tk in m:
        if tk != preferredbackend:
            for format in m[tk].outformats:
                if format not in outformat_backends:
                    outformat_backends[format] = tk


format_backends = {}
outformat_backends = {}
set_format_backends(backend)
set_outformat_backends(backend)

informats = ''
for format in format_backends:
    informats += "*.%s " % format
    if format_backends[format] == 'pybel':
        for compression in ('gz', 'tar', 'bz', 'bz2', 'tar.gz', 'tar.bz', 'tar.bz2'):
            informats += "*.%s.%s " % (format, compression)

_debug('%s is the default backend' % backend)

# Some default values:

HBA_t = 2
HBD_t = 1
ClogP_t = Decimal(1)  # 1.5
tanimoto_t = Decimal('0.75')
tanimoto_d = Decimal('0.9')
MW_t = 25
RB_t = 1
mind = 36
maxd = 50

# Dict of ZINC subsets
ZINC_subsets = {
    "lead-like": "1"
    , "fragment-like": "2"
    , "drug-like": "3"
    , "all-purchasable": "6"
    , "everything": "10"
    , "clean-leads": "11"
    , "clean-fragments": "12"
    , "clean-drug-like": "13"
    , "all-clean": "16"
    , "leads-now": "21"
    , "frags-now": "22"
    , "drugs-now": "23"
    , "all-now": "26"
    , "sarah": "37"
    , "Stan": "94"
}

calc_functs = {
    'pybel': {
        'calc_hba': lambda mol: mol.calcdesc(['HBA2'])['HBA2']
        , 'calc_hbd': lambda mol: mol.calcdesc(['HBD'])['HBD']
        , 'calc_clogp': lambda mol: Decimal(str(mol.calcdesc(['logP'])['logP']))
        , 'calc_rot': lambda mol: mol.OBMol.NumRotors()
    }
    , 'cdk': {
        'calc_hba': lambda mol: mol.calcdesc(['hBondacceptors'])['hBondacceptors']
        , 'calc_hbd': lambda mol: mol.calcdesc(['hBondDonors'])['hBondDonors']
        , 'calc_clogp': lambda mol: Decimal(str(mol.calcdesc(['xlogP'])['xlogP']))
        , 'calc_rot': lambda mol: mol.calcdesc(['rotatableBondsCount'])['rotatableBondsCount']
    }
    , 'rdk': {
        'calc_hba': lambda mol: mol.calcdesc(['NumHAcceptors'])['NumHAcceptors']
        , 'calc_hbd': lambda mol: mol.calcdesc(['NumHDonors'])['NumHDonors']
        , 'calc_clogp': lambda mol: Decimal(str(mol.calcdesc(['MolLogP'])['MolLogP']))
        , 'calc_rot': lambda mol: mol.calcdesc(['NumRotatableBonds'])['NumRotatableBonds']
    }
}


class ComparableMol(object):
    """
    """

    def __init__(self, mol, bknd):
        self.mol = mol
        self.__dict__['mol_' + bknd] = mol

    # Calculate all interesting descriptors. Called only when needed

    def calc_hba(self, mol, b):
        return calc_functs[b]['calc_hba'](mol)

    def calc_hbd(self, mol, b):
        return calc_functs[b]['calc_hbd'](mol)

    def calc_clogp(self, mol, b):
        return calc_functs[b]['calc_clogp'](mol)

    def calc_rot(self, mol, b):
        return calc_functs[b]['calc_rot'](mol)

    def calc_fp(self, mol, b):
        return ComFp(fp=mol.calcfp(FPTYPE))

    def calc_mw(self, mol, b):
        return mol.molwt

    def calc_title(self, mol, b):
        return mol.title

    def calc_can(self, mol, b):
        try:
            can = mol.write(REP)
        except Exception as e:
            _debug(e)
            return None
        if REP in ('smi', 'can') and b == 'pybel':
            can = can.split('\t')[0]
        elif REP == 'inchikey':
            can = can[:25]
        return can

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            if attr == 'mol':
                mol = None
                b = backend
            elif attr.startswith('mol'):
                b = attr.split('_')[1]
                self.__dict__[attr] = m[b].Molecule(self.mol)
            else:
                if attr in backenddict:
                    b = backenddict[attr]
                else:
                    b = backend
                bmol = 'mol_' + b
                if bmol not in self.__dict__:
                    mol = m[b].Molecule(self.mol)
                    self.__dict__[bmol] = mol
                else:
                    mol = self.__dict__[bmol]
                self.__dict__[attr] = eval('self.calc_%s' % attr)(mol, b)
        return self.__dict__[attr]

    def __str__(self):
        """
        For debug purposes
        """
        return "Title: %s; HBA: %s; HBD: %s; CLogP: %s; MW:%s \n" % (
        self.title, self.hba, self.hbd, self.clogp, self.mw)


class DbMol(ComparableMol):
    """
    Molecule from a database, with precalculated descriptors
    """

    def __init__(self, row):
        self.inchikey, maccsbits, self.rot, self.mw, self.clogp, self.hba, self.hbd, self.mdlmol = row
        self.bitset = set(eval(maccsbits))
        if REP == 'inchikey':
            self.can = self.inchikey

    def calc_fp(self, *args):
        return ComFp(bitset=self.bitset)

    def calc_mol(self, bmol, b):
        # Check wether it's compressed
        if self.mdlmol[-4:-1] == 'END':
            return self.mdlmol
        return m[b].readstring('mol', str(zlib.decompress(self.mdlmol)))


class ComFp(object):
    """
    Comparable fingerprint from a set of bits
    """

    def __init__(self, bitset=None, fp=None):
        self.fp = fp
        if bitset:
            self.bits = bitset
        elif fp:
            self.bits = set(fp.bits)
        if backend == 'cdk':
            self.bits = set([bit + 1 for bit in self.bits])

    def __or__(self, other):
        """
        This is borrowed from cinfony's webel.py
        Returns the Tanimoto score between two sets of bits
        """
        if self.fp and other.fp:
            return self.fp | other.fp
        else:
            return len(self.bits & other.bits) / float(len(self.bits | other.bits))

    def __str__(self):
        return ", ".join([str(x) for x in self.bits])


def get_zinc_slice(slicename='all', subset='10', cachedir=tempfile.gettempdir(), keepcache=False):
    """
    returns an iterable list of files from  online ZINC slices
    """
    if slicename in ('all', 'single', 'usual', 'metals'):
        script = "http://zinc12.docking.org/db/bysubset/%s/%s.sdf.csh" % (subset, slicename)
        _debug('Downloading files in %s' % script)
        handler = urllib.request.urlopen(script)
        _debug("Reading ZINC data...")
        scriptcontent = handler.read().split('\n')
        handler.close()
        filelist = []
        parenturl = None
        for line in scriptcontent:
            if not line.startswith('#'):
                if not parenturl and 'http://' in line:
                    parenturl = 'http://' + line.split('http://')[1].split()[0]
                    if not parenturl.endswith('/'):
                        parenturl += '/'
                elif line.endswith('.sdf.gz'):
                    filelist.append(line)
        yield len(filelist)
        random.shuffle(filelist)
        for file in filelist:
            dbhandler = urllib.request.urlopen(parenturl + file)
            outfilename = os.path.join(cachedir, file)
            download_needed = True
            if keepcache:
                filesize = dbhandler.info().get('Content-Length')
                if filesize:
                    filesize = int(filesize)

                if os.path.isfile(outfilename):
                    localsize = os.path.getsize(outfilename)
                    download_needed = localsize != filesize
                    if download_needed:
                        _debug("Local file outdated or incomplete")

            if download_needed:
                _debug('Downloading %s' % parenturl + file)
                outfile = open(outfilename, "wb")
                outfile.write(dbhandler.read())
                outfile.close()
            else:
                _debug("Loading cached file: %s" % outfilename)
            dbhandler.close()
            yield str(outfilename)

            if not keepcache:
                try:
                    os.remove(outfilename)
                except Exception as e:
                    _debug("Unable to remove %s" % (outfilename))
                    _debug(str(e))
    else:
        raise Exception("Unknown slice")


def get_fileformat(filename):
    """
    Guess the file format from its extension
    """
    index = -1
    ext = filename.split(".")[index].lower()
    while ext in ('gz', 'tar', 'bz', 'bz2'):
        index -= 1
        ext = filename.split(".")[index].lower()
    if ext in format_backends:
        return ext
    else:
        _debug("%s: unknown format" % filename)
        raise ValueError


def get_format_backend(filename):
    format = get_fileformat(filename)
    ext = filename.split(".")[1].lower()
    if ext not in ('gz', 'tar', 'bz', 'bz2'):
        return format_backends[format]
    else:
        return 'pybel'


def query_db(conn, table='Molecules'):
    """
    Parses files from a SQL database with dbapi 2.0
    """
    cursor = conn.cursor()
    if REP in ('can', 'smi'):
        f = 'smiles'
    else:
        f = REP
    cursor.execute(
        """SELECT `%s`,`maccs`, `rotatable_bonds`, `weight`, `logp`, `hba`, `hbd`, `mol` FROM %s;""" % (f, table))
    rowcount = 0
    for row in cursor:
        rowcount += 1
        try:
            mol = DbMol(row)
            yield mol, rowcount, 'database'
        except Exception as e:
            print(e)
    else:
        cursor.close()


def _parse_db_files(filelist):
    """
    Parses files where to look for decoys
    """
    filecount = 0
    if type(filelist) == list:
        random.shuffle(filelist)
    for dbfile in filelist:
        b = get_format_backend(dbfile)
        mols = m[b].readfile(get_fileformat(dbfile), dbfile)
        for mol in mols:
            if mol:
                try:
                    cmol = ComparableMol(mol, b)
                    yield cmol, filecount, dbfile
                except Exception as e:
                    _debug(e)
        filecount += 1


def parse_query_files(filelist):
    """
    Parses files containing active ligands
    """
    b = 'pybel'
    query_dict = {}
    with open(filelist[0]) as mol_f:
        for s in mol_f:
            mol = pybel.readstring('smi', s)
            if mol:
                try:
                    cmol = ComparableMol(mol, b)
                    query_dict[cmol] = 0
                except Exception as e:
                    _debug(e)
    return query_dict


def parse_db_files(filelist):
    """
    Parses files containing active ligands
    """
    b = 'pybel'
    query_dict = {}
    with open(filelist[0]) as mol_f:
        for s in mol_f:
            try:
                mol = pybel.readstring('smi', s.split(",")[1])
            except:
                continue
            if mol:
                try:
                    cmol = ComparableMol(mol, b)
                    yield cmol, 1, filelist[0]
                except Exception as e:
                    _debug(e)


def parse_decoy_files(decoyfilelist):
    """
    Parses files containing known decoys
    """
    decoy_set = set()
    for decoyfile in decoyfilelist:
        decoyfile = str(decoyfile)
        b = get_format_backend(decoyfile)
        mols = m[b].readfile(get_fileformat(decoyfile), decoyfile)
        for mol in mols:
            if mol:
                try:
                    cmol = ComparableMol(mol, b)
                    decoy_set.add(cmol)
                except Exception as e:
                    _debug(e)
    return decoy_set


def isdecoy(
        db_mol
        , ligand
        , HBA_t=HBA_t
        , HBD_t=HBD_t
        , ClogP_t=ClogP_t
        , MW_t=MW_t
        , RB_t=RB_t
):
    """
    Check if db_mol can be considered a decoy of ligand
    """
    if ligand.hbd - HBD_t <= db_mol.hbd <= ligand.hbd + HBD_t:
        if ligand.mw - MW_t <= db_mol.mw <= ligand.mw + MW_t:
            if ligand.rot - RB_t <= db_mol.rot <= ligand.rot + RB_t:
                if ligand.hba - HBA_t <= db_mol.hba <= ligand.hba + HBA_t:
                    if ligand.clogp - ClogP_t <= db_mol.clogp <= ligand.clogp + ClogP_t:
                        return True
    return False


def get_ndecoys(ligands_dict, maxd):
    return sum((x for x in ligands_dict.values() if not maxd or maxd >= x))


def checkoutputfile(outputfile):
    """
    Return a safe output filename
    """
    fileexists = 0
    if os.path.splitext(outputfile)[1].lower()[1:] not in m[backend].outformats:
        outputfile += "_decoys.sdf"
    while os.path.isfile(outputfile):
        fileexists += 1
        filename, extension = os.path.splitext(outputfile)
        if filename.endswith("_%s" % (fileexists - 1)):
            filename = '_'.join(filename.split('_')[:-1]) + "_%s" % fileexists
        else:
            filename += "_%s" % fileexists
        outputfile = filename + extension
    return outputfile


def find_decoys(
        query_files=[]
        , db_files=[]
        , outputfile='found_decoys'
        , HBA_t=HBA_t
        , HBD_t=HBD_t
        , ClogP_t=ClogP_t
        , tanimoto_t=tanimoto_t
        , tanimoto_d=tanimoto_d
        , MW_t=MW_t
        , RB_t=RB_t
        , mind=mind
        , maxd=maxd
        , decoy_files=[]
        , stopfile=''
        , unique=False
        , internal=REP
        , conn=None
        , fptype='MACCS'
        , toolkits={}
        , default_toolkit=backend
):
    """
    This is the star of the show
    """
    internal = internal.lower().strip()
    if default_toolkit:
        global backend
        backend = default_toolkit
        set_format_backends(backend)
        set_outformat_backends(backend)
        for desc in backenddict:
            backenddict[desc] = backend
    if fptype:
        global FPTYPE
        FPTYPE = fptype.upper()
        set_fp_backend()
    if internal in outformat_backends:
        global REP
        REP = internal
        backenddict['can'] = outformat_backends[REP]
    else:
        _debug('Unrecognized format:%s' % internal)
        _debug('Using default:%s' % REP)
    if toolkits:
        for desc in toolkits:
            backenddict[desc] = toolkits[desc]
    _debug('Using %s for internal indexation' % REP)
    outputfile = checkoutputfile(outputfile)
    tanimoto_t = Decimal(str(tanimoto_t))
    tanimoto_d = Decimal(str(tanimoto_d))
    ClogP_t = Decimal(str(ClogP_t))
    _debug("Looking for decoys!")

    db_entry_gen = parse_db_files(db_files)

    if conn:
        try:
            db_entry_gen = itertools.chain(query_db(conn), db_entry_gen)
        except Exception as e:
            _debug(e)

    used_db_files = set()

    ligands_dict = parse_query_files(query_files)

    nactive_ligands = len(ligands_dict)

    complete_ligand_sets = 0

    minreached = False
    if mind:
        total_min = nactive_ligands * mind
        yield ('total_min', total_min, nactive_ligands)
    else:
        mind = None

    decoys_can_set = set()
    ndecoys = get_ndecoys(ligands_dict, maxd)
    ligands_max = set()

    outputfile = checkoutputfile(outputfile)
    format = get_fileformat(outputfile)
    outbackend = format_backends[format]
    decoyfile = m[outbackend].Outputfile(format, str(outputfile))
    decoys_fp_set = set()

    yield ('ndecoys', ndecoys, complete_ligand_sets)

    for db_mol, filecount, db_file in db_entry_gen:
        saved = False
        used_db_files.add(db_file)
        yield ('file', filecount, db_file)
        if maxd and len(ligands_max) >= nactive_ligands:
            _debug('Maximum reached')
            minreached = True
            break
        if complete_ligand_sets >= nactive_ligands:
            _debug('All decoy sets complete')
            break
        if not mind or ndecoys < total_min:
            ligands_decoy = set()
            try:
                for ligand in ligands_dict:
                    if ligand not in ligands_max:
                        if isdecoy(db_mol, ligand, HBA_t, HBD_t, ClogP_t, MW_t, RB_t):
                            ligands_decoy.add(ligand)
                if not ligands_decoy:
                    continue
                too_similar = False
                for active in ligands_dict:
                    active_T = active.fp | db_mol.fp
                    if active_T > tanimoto_t:
                        too_similar = True
                        break
                if not too_similar:
                    if db_mol.can in decoys_can_set:
                        continue
                    if tanimoto_d < 1:
                        for decoyfp in decoys_fp_set:
                            decoy_T = decoyfp | db_mol.fp
                            if decoy_T > tanimoto_d:
                                too_similar = True
                                break
                    if too_similar:
                        continue
                    for ligand in ligands_decoy:
                        if maxd and ligands_dict[ligand] >= maxd:
                            ligands_max.add(ligand)
                            continue
                        ligands_dict[ligand] += 1
                        if not saved:
                            decoyfile.write(db_mol.__getattribute__('mol_' + outbackend))
                            decoys_can_set.add(db_mol.can)
                            decoys_fp_set.add(db_mol.fp)
                            saved = True
                        ndecoys = get_ndecoys(ligands_dict, maxd)
                        _debug('%s decoys found' % ndecoys)
                        if ligands_dict[ligand] == mind:
                            _debug('Decoy set completed for ' + ligand.title)
                            complete_ligand_sets += 1
                        yield ('ndecoys', ndecoys, complete_ligand_sets)
                        if unique:
                            break
            except Exception as e:
                _debug(e)
        else:
            _debug("finishing")
            break
        if os.path.exists(stopfile):
            os.remove(stopfile)
            _debug('stopping by user request')
            break
    else:
        _debug('No more input molecules')

    if mind:
        _debug('Completed %s of %s decoy sets' % (complete_ligand_sets, nactive_ligands))
        minreached = complete_ligand_sets >= nactive_ligands
    if minreached:
        _debug("Found all wanted decoys")
    else:
        _debug("Not all wanted decoys found")
    # Generate logfile
    # log = '"%s %s log file generated on %s"\n' % (metadata.NAME, metadata.VERSION, datetime.datetime.now())
    log = ""
    log += "\n"
    log += '"Output file:","%s"\n' % outputfile
    log += "\n"
    log += '"Active ligand files:"\n'
    for file in query_files:
        log += '"%s"\n' % str(file)
    log += "\n"
    log += '"Decoy sources:"\n'
    for file in used_db_files:
        log += '"%s"\n' % str(file)
    log += "\n"
    log += '"Search settings:"\n'
    log += '"Active ligand vs decoy tanimoto threshold","%s"\n' % str(tanimoto_t)
    log += '"Decoy vs decoy tanimoto threshold","%s"\n' % str(tanimoto_d)
    log += '"Hydrogen bond acceptors range","%s"\n' % str(HBA_t)
    log += '"Hydrogen bond donors range","%s"\n' % str(HBD_t)
    log += '"LogP range","%s"\n' % str(ClogP_t)
    log += '"Molecular weight range","%s"\n' % str(MW_t)
    log += '"Rotational bonds range","%s"\n' % str(RB_t)
    log += '"Minimum number of decoys per active ligand","%s"\n' % str(mind)
    log += '"Maximum number of decoys per active ligand","%s"\n' % str(maxd)
    log += "\n"
    log += '"Active ligand","HBA","HBD","logP","MW","RB","number of Decoys found"\n'
    for active in ligands_dict:
        log += '"%s","%s","%s","%s","%s","%s","%s"\n' % tuple([str(f) for f in (
        active.title, active.hba, active.hbd, active.clogp, active.mw, active.rot, ligands_dict[active])])
    log += "\n"

    # print(log)
    # logfile = open('%s_log.csv' % outputfile,  'wb')
    # logfile.write(log)
    # logfile.close()

    decoyfile.close()

    if not decoys_fp_set:
        if os.path.exists(outputfile):
            os.remove(outputfile)
    _debug(backenddict)
    # Last, special yield:
    yield ('result', ligands_dict, [outputfile, minreached])
