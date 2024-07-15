import sys
from functools import wraps
import inspect

def variable_guard(func):
    """ Decorator to guard against variable leakage in the calling frame."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the calling frame
        frame = inspect.currentframe().f_back
        
        # Save local and global variables
        saved_locals = dict(frame.f_locals)
        saved_globals = dict(frame.f_globals)
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore global variables
            current_globals = frame.f_globals
            for name in list(current_globals.keys()):
                if not name.startswith('__'):
                    if name not in saved_globals:
                        del current_globals[name]
                    else:
                        current_globals[name] = saved_globals[name]
            
            # Restore local variables
            current_locals = frame.f_locals
            for name in list(current_locals.keys()):
                if name not in saved_locals:
                    del current_locals[name]
                else:
                    current_locals[name] = saved_locals[name]
            
            # Update the frame's locals
            frame.f_locals.update(current_locals)

    return wrapper