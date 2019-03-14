import os

flags = [
'-Wall',
'-Wextra',
'-Werror',
'-std=c++14',
'-x', 'c++',
'-isystem', '/usr/include/clang/7/include',
'-isystem', '/usr/include/clang/6/include',
'-I', 'src/lib',
]

DIR_OF_THIS_SCRIPT = os.path.abspath( os.path.dirname( __file__ ) )


# First are functions that seem necessary for older YCM.

def MakeRelativePathsInFlagsAbsolute( flags, working_directory ):
  if not working_directory:
    return list( flags )
  new_flags = []
  make_next_absolute = False
  path_flags = [ '-isystem', '-I', '-iquote', '--sysroot=' ]
  for flag in flags:
    new_flag = flag
    if make_next_absolute:
      make_next_absolute = False
      if not flag.startswith( '/' ):
        new_flag = os.path.join( working_directory, flag )
    for path_flag in path_flags:
      if flag == path_flag:
        make_next_absolute = True
        break
      if flag.startswith( path_flag ):
        path = flag[ len( path_flag ): ]
        new_flag = path_flag + os.path.join( working_directory, path )
        break
    if new_flag:
      new_flags.append( new_flag )
  return new_flags

# This seems to be the entry point for older YCM; this function is called by
# ycmd to produce flags for a file.
def FlagsForFile( filename, **kwargs ):
  final_flags = MakeRelativePathsInFlagsAbsolute(flags, DIR_OF_THIS_SCRIPT)
  return {
    'flags': final_flags,
  }


# Second are functions that seem necessary for newer YCM.

def IsHeaderFile( filename ):
  extension = os.path.splitext( filename )[ 1 ]
  return extension in [ '.h', '.hxx', '.hpp', '.hh' ]

SOURCE_EXTENSIONS  = [ '.cpp', '.cxx', '.cc', '.c' ]

def FindCorrespondingSourceFile( filename ):
  if IsHeaderFile( filename ):
    basename = os.path.splitext( filename )[ 0 ]
    for extension in SOURCE_EXTENSIONS:
      replacement_file = basename + extension
      if os.path.exists( replacement_file ):
        return replacement_file
  return filename

# This seems to be the entry point for newer YCM.
def Settings( **kwargs ):
  if kwargs[ 'language' ] == 'cfamily':
    filename = FindCorrespondingSourceFile( kwargs[ 'filename' ] )
    return {
      'flags': flags,
      'include_paths_relative_to_dir': DIR_OF_THIS_SCRIPT,
      'override_filename': filename
    }
  return {}
