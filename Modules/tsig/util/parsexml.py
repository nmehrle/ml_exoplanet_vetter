#
# Copyright (C) 2010 - Martin Owens <doctormo@gmail.com>
#               2017 - Massachusetts Institute of Technology (MIT)
#
# This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>
#
# Originally taken from pypi validator package (combined)
#
"""
Parse an XML file into a data structure and allow validation
"""

import re
import os
import logging

import xml.sax
from xml.sax.handler import ContentHandler
from datetime import datetime, time, date

def islist(l):
    if not isinstance(l, (list, tuple)):
        return [l]
    return l

# Logic Modes
MODE_OR  = False
MODE_AND = True

class NoData(ValueError):
    """Report error about there not being any data"""

class NoRootDocument(ValueError):
    """Report error concerning the lack of root"""

class NoTypeFound(KeyError):
    """Reported when there the named type is not found"""

class ElementErrors(dict):
    """Keep track of errors as they're added, true if errors."""
    def __init__(self, mode=MODE_AND):
        self._in_error = 0
        self._added = 0
        self._mode = mode

    def __setitem__(self, key, value):
        if key in self:
            self.remove_error(self[key])
        super(ElementErrors, self).__setitem__(key, value)
        self.add_error(value)

    def __repr__(self):
        if self._added and not self._in_error:
            return "NO_ERROR"
        return super(ElementErrors, self).__repr__()

    def pop(self, key):
        self.remove_error(super(ElementErrors, self).pop(key))

    def update(self, errors):
        """Merge in errors from a seperate validation process"""
        # Adding a batch of errors is counted as one.
        self.add_error(errors)
        if isinstance(errors, ElementErrors):
            if errors._mode == MODE_OR and not errors:
                errors = dict((a,b) for (a,b) in errors.items() if b == NO_ERROR)
            else:
                errors = dict((a,b) for (a,b) in errors.items() if b != INVALID_EXIST)
        super(ElementErrors, self).update(errors)

    def add_error(self, error):
        if error:
            self._in_error += 1
        self._added += 1

    def remove_error(self, error):
        if error:
            self._in_error -= 1
        self._added -= 1

    def __nonzero__(self):
        #print "In ERROR: %s %i:%i" % (str(self._data), self._in_error, self._added)
        if self._mode == MODE_OR:
            if self._added > 0:
                return self._in_error >= self._added
            return False
        return self._in_error != 0

    def __bool__(self):
        "PY3 for nonzero"
        return self.__nonzero__()

    def __eq__(self, errors):
        if not isinstance(errors, dict):
            return False
            #raise ValueError("Can't compare error dictionary with %s" % type(errors))
        for (key, value) in super(ElementErrors, self).items():
            if key not in errors or errors[key] != value:
                return False
        return True

    def __ne__(self, opt):
        return not self.__eq__(opt)


class ValidateError(object):
    """Control the validation errrors and how they're displayed"""
    def __init__(self, code, msg, desc=None):
        self._code = code
        self._msg  = msg
        self._desc = desc
        self._cont = None

    def __repr__(self):
        return self._msg.upper().replace(' ','_')
        # Used for debugging, maybe enable it in a future version
        if self._cont:
            return "%s (%s)\n" % (self._msg, str(self._cont))
        return "#%d %s (%s)\n" % (self._code, self._msg, self._desc)

    def __call__(self, context):
        """When an error is called, it adds context to it"""
        if not self._cont:
            self._cont = context
        return self

    def __int__(self):
        return self._code

    def __str__(self):
        result = [ self._msg ]
        if self._desc:
            result.append( self._desc )
        return ' '.join(result)

    def __unicode__(self):
        return unicode(str(elf._msg))

    def __nonzero__(self):
        return self._code > 0

    def __bool__(self):
        "PY3 for nonzero"
        return self.__nonzero__()

    def __eq__(self, opt):
        if isinstance(opt, ValidateError):
            opt = opt._code
        return self._code == opt

    def __ne__(self, opt):
        return not self.__eq__(opt)



# Validation Error codes
NO_ERROR            = ValidateError(0x00, 'No Error')
INVALID_TYPE        = ValidateError(0x01, 'Invalid Node Type')
INVALID_PATTERN     = ValidateError(0x02, 'Invalid Pattern', 'Regex Pattern failed')
INVALID_MINLENGTH   = ValidateError(0x03, 'Invalid MinLength', 'Not enough nodes present')
INVALID_MAXLENGTH   = ValidateError(0x04, 'Invalid MaxLength', 'Too many nodes present')
INVALID_MATCH       = ValidateError(0x05, 'Invalid Match', 'Node to Node match failed')
INVALID_VALUE       = ValidateError(0x06, 'Invalid Value', 'Fixed string did not match')
INVALID_NODE        = ValidateError(0x07, 'Invalid Node', 'Required data does not exist for this node')
INVALID_ENUMERATION = ValidateError(0x08, 'Invalid Enum', 'Data not equal to any values supplied')
INVALID_MIN_RANGE   = ValidateError(0x09, 'Invalid Min Range', 'Less than allowable range')
INVALID_MAX_RANGE   = ValidateError(0x0A, 'Invalid Max Range', 'Greater than allowable range')
INVALID_NUMBER      = ValidateError(0x0B, 'Invalid Number', 'Data is not a real number')
INVALID_COMPLEX     = ValidateError(0x0C, 'Invalid Complex', 'Failed to validate Complex Type')
INVALID_REQUIRED    = ValidateError(0x0D, 'Invalid Required', 'Data is required, but missing.')
INVALID_EXIST       = ValidateError(0x0E, 'Invalid Exist', 'This data shouldn\'t exist.')
INVALID_MIN_OCCURS  = ValidateError(0x0F, 'Invalid Occurs', 'Minium number of occurances not met')
INVALID_MAX_OCCURS  = ValidateError(0x10, 'Invalid Occurs', 'Maxium number of occurances exceeded')
INVALID_XPATH       = ValidateError(0x11, 'Invalid XPath', 'The path given doesn\'t exist.')

# When python goes wrong
CRITICAL = ValidateError(0x30, 'Critical Problem')

# Custom internal methods for checking values
INVALID_CUSTOM = ValidateError(0x40, 'Invalid Custom', 'Custom filter method returned false')

# Extra Error codes
INVALID_DATE_FORMAT = ValidateError(0x50, 'Invalid Date Format', 'Format of date can\'t be parsed')
INVALID_DATE        = ValidateError(0x51, 'Invalid Date', 'Date is out of range or otherwise not valid')


class ParseXML(object):
    """Create a new xml parser"""
    def __init__(self, xml):
        self._data = None
        self._xml = xml

    @property
    def data(self):
       """Return the parsed data structure."""
       if not self._data:
           parser = CustomParser()
           xml.sax.parse(self._xml, parser)
           self._data = parser._root
       return self._data

    @property
    def definition(self):
        """Convert the data into a definition, assume it's in xsd format."""
        data   = self.data.get('schema', None)
        result = {}
        if not data:
            return result
        result['complexTypes'] = self._complexes(data.get('complexType', None))
        result['simpleTypes'] = self._simples(data.get('simpleType', None))
        result['root'] = self._elements(data['element'], [])
        return result

    def _complexes(self, data):
        if not data:
            return {}
        complexes = {}
        for item in islist(data):
            if '_name' in item:
                complexes[item['_name']] = self._complex( item )
            else:
                logging.warn("Complex type without a name!")
        return complexes

    def _complex(self, data, extend=False):
        elements = []
        if 'element' in data:
            self._elements( data['element'], elements )
        if 'attribute' in data:
            self._elements( data['attribute'], elements, '_' )
        # Logical And/Or have to be added as arrays of complexes
        for item in islist(data.get('or', [])):
            result = self._complex( item, extend=True )
            if extend:
                elements.extend( result )
            else:
                elements.append( result )
        # Elements can be extended or appended depending on their logic.
        for item in list(data.get('and', [])):
            result = self._complex( item, extend=False )
            if extend:
                elements.extend( result )
            else:
                elements.append( result )
        # Politly return the elements configuration.
        return elements

    def _simples(self, data):
        if not data:
            return {}
        simples = {}
        name = data.pop('_name')
        simples[name] = self._simple( name, data )
        return simples

    def _simple(self, name, data):
        for key in data.keys():
            if key[1] == '_':
                data[key[1:]] = data.pop(key)
        return data

    def _elements(self, data, res, prefix=''):
        if not isinstance(data, list):
            data = [ data ]
        for element in data:
            res.append(self._element( element, prefix ))
        return res

    def _element(self, data, prefix=''):
        element = {}
        if 'complexType' in data:
            element['complexType'] = self._complex( data.pop('complexType') )
        elif '_type' in data:
            # Get rid of the namespace, we don't need it.
            element['type'] = data.pop('_type').split(':')[-1]
        # Translate the rest of the element's attributes
        for key in data.keys():
            if key in ('name','_name'):
                element['name'] = prefix + data[key]
            elif key[0] == '_':
                element[key[1:]] = data[key]
        return element



class CustomParser(ContentHandler):
    """Parsing the xml via SAX."""
    def __init__(self):
        """Create a new parser object."""
        self._root    = {}
        self._current = self._root
        self._parents = [ self._root ]
        self._count   = 0
        super(type(self), self).__init__()

    def skippedEntity(self, name):
        """Any elements that are skipped"""
        logging.warn("Skipping element %s" % name)

    def startElement(self, name, atrs):
        """Start a new xml element"""
        name = name.split(':')[-1]
        c    = self._current
        new  = {}

        if name not in c or not c[name]:
            c[name] = new
        else:
            if not isinstance(c[name], list):
                c[name] = [ c[name] ]
            c[name].append( new )

        self._parents.append(c)
        self._count += 1
        self._name = name
        self._current = new

        for key in atrs.keys():
            value = atrs[key]
            key = key.split(':')
            if key[0] != 'xmlns':
                self._current['_' + key[-1]] = value

    def endElement(self, name):
        """Ends an xml element"""
        name = name.split(':')[-1]
        self._count += 1
        self._current = self._parents.pop()

    def characters(self, text):
        """Handle part of a cdata by concatination"""
        t = text
        # There must be a better way to test char exists.
        if t.strip():
            p = self._parents[-1]
            c = p.get(self._name, '')
            if isinstance(c, dict):
                if c:
                    if '+data' in c:
                        c['+data'] += t
                    else:
                        c['+data'] = t

                else:
                    p[self._name] = t

            elif isinstance(c, list):
                # Remove an empty end item
                if isinstance(c[-1], dict) and not c[-1]:
                    c.pop()
                c.append(t)
            elif self._name in p:
                p[self._name] += t
            else:
                p[self._name] = t


def test_datetime(data, stype=None):
    """Test to make sure it's a valid datetime"""
    try:
        if '-' in data and ':' in data:
            datetime.strptime(data, "%Y-%m-%d %H:%M:%S")
        elif '-' in data:
            datetime.strptime(data, "%Y-%m-%d")
        elif ':' in data:
            datetime.strptime(data, "%H:%M:%S")
        else:
            return INVALID_DATE_FORMAT
    except:
        return INVALID_DATE
    return NO_ERROR

#Primitive types: [ anyURI, base64Binary, boolean, date,
#  dateTime, decimal, double, duration, float, hexBinary,
#  gDay, gMonth, gMonthDay, gYear, gYearMonth, NOTATION,
#  QName, string, time ]

BASE_CLASSES = {
    'complexTypes': {},
    'simpleTypes': {
        'string'    : { 'pattern' : r'.*' },
        'integer'   : { 'pattern' : r'[\-]{0,1}\d+' },
        'index'     : { 'pattern' : r'\d+' },
        'double'    : { 'pattern' : r'[0-9\-\.]*' },
        'token'     : { 'base'    : r'string', 'pattern' : '\w+' },
        'boolean'   : { 'pattern' : r'1|0|true|false' },
        'email'     : { 'pattern' : r'.+@.+\..+' },
        'date'      : { 'pattern' : r'\d\d\d\d-\d\d-\d\d', 'base' : 'datetime' },
        'time'      : { 'pattern' : r'\d\d:\d\d:\d\d',     'base' : 'datetime' },
        'datetime'  : { 'pattern' : r'(\d\d\d\d-\d\d-\d\d)?[T ]?(\d\d:\d\d:\d\d)?', 'custom' : test_datetime },
        'percentage': { 'base'    : r'double', 'minInclusive' : 0, 'maxInclusive' : 100 },
        'exponent'  : { 'pattern': r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?' }
    }
}


class Validator(object):
    """Validation Machine, parses data and outputs error structures.

    validator = Validator(definition, strict_root, strict_values)

    definition   - Validation structure (see main documentation)
    strict_root  - Don't automatically add a root element dictionary.
    strict_exist - Add errors for elements and attributes not in the schema.

    """
    def __init__(self, definition, strict_root=False, strict_exist=True, debug=False):
       self._strict_root  = strict_root
       self._strict_exist = strict_exist
       self._definition = None
       self._debug = debug
       if isinstance(definition, str):
           definition = self._load_file(definition, True)
       self.setDefinition( definition )

    def validate(self, data):
        """
        Validate a set of data against this validator.
        Returns an errors structure or 0 if there were no errors.
        """
        if isinstance(data, str):
            data = self._load_file( data )
        d = self._definition
        # Save the root data for this validation so it can be
        # used for xpath queries later on in the validation.
        self.current_root_data = data

        # Sometimes we want to be lazy, allow us to be.
        if 'root' not in d and not self._strict_root:
            d = { 'root' : d }
        elif not self._strict_root:
            if data != None:
                return self._validate_elements( d['root'], data )
            else:
                raise NoData()
        else:
            raise NoRootDocument()

    def setDefinition(self, definition):
        """Set the validators definition, will load it (used internally too)"""
        self._definition = self._load_definition( definition )

    def getErrorString(self, err):
        """Return a human readable string for each error code."""
        if err > 0 and err <= len(ERRORS):
            return ERRORS[err]
        return 'Invalid error code'

    def _load_definition(self, definition):
        """Internal method for loading a definition into the validator."""
        # Make sure we have base classes in our definition.
        self._update_types(definition, BASE_CLASSES)

        # Now add any includes (external imports)
        for filename in definition.get('include', []):
            include = None
            if type(filename) in (str, unicode):
                include = self._load_definition_from_file( filename )
            elif type(filename) == dict:
                include = filename
            if include:
                self._update_types(definition, include)
            else:
                raise Exception("Can't load include: %s" % str(filename))
        return definition

    def _update_types(self, definition, source):
        """Update both simple and compelx types."""
        self._update_type(definition, source, 'simpleTypes')
        self._update_type(definition, source, 'complexTypes')

    def _update_type(self, definition, source, ltype):
        """This merges the types together to get a master symbol table."""
        if not definition:
            raise ValueError("Definition not defined!")
        definition[ltype] = definition.get(ltype, {})
        definition[ltype].update(source.get(ltype, {}))

    def _load_definition_from_file(self, filename):
        """Internal method for loading a definition from a file"""
        return self._load_definition( self._load_file( filename ) )

    def _validate_elements(self, definition, data, mode=MODE_AND, primary=True):
        """Internal method for validating a list of elements"""
        errors = ElementErrors(mode)

        # This should be AND or OR and controls the logic flow of the data varify
        if mode not in (MODE_AND, MODE_OR):
            raise Exception("Invalid mode '%s', should be MODE_AND or MODE_OR." % mode)
  
        if not isinstance(definition, list):
            raise Exception("Definition is not in the correct format: expected list (got %s)." % type(definition))
        
        for element in definition:
            # Element data check
            if isinstance(element, dict):
                name = element.get('name', None)
                # Skip element if it's not defined
                if not name:
                    logging.warn("Skipping element, no name")
                    continue
                # We would pass in only the data field selected, but we need everything.
                errors[name] = self._validate_element( element, data, name )
            elif isinstance(element, list):
                errors.update(self._validate_elements( element, data, not mode, False ))
            else:
                logging.warn("This is a complex type, but isn't element.")

        # These are all the left over names
        if self._strict_exist and primary:
            for name in data.keys():
                if name not in errors:
                    errors[name] = INVALID_EXIST

        return errors


    def _validate_element(self, definition, all_data, name):
        """Internal method for validating a single element"""
        results = []
        proped  = False

        data = all_data.get(name, None)
        if data != None and not isinstance(data, list):
            proped = True
            data   = [ data ]

        minOccurs = int(definition.get('minOccurs', 1))
        maxOccurs =     definition.get('maxOccurs', 1)
        dataType  =     definition.get('type',      'string')
        fixed     =     definition.get('fixed',     None)
        default   =     definition.get('default',   None)

        # minOccurs checking
        if minOccurs >= 1:
           if data != None:
               if minOccurs > len(data):
                   return INVALID_MIN_OCCURS
           elif default != None:
               data = [ default ]
           else:
               return INVALID_REQUIRED
           if maxOccurs not in [ None, 'unbounded' ] and int(maxOccurs) < minOccurs:
               maxOccurs = minOccurs
        elif data == None:
            # No data and it wasn't required
            return NO_ERROR

        # maxOccurs Checking
        if maxOccurs != 'unbounded':
            if int(maxOccurs) < len(data):
                return INVALID_MAX_OCCURS
    
        for element in data:
            # Fixed checking
            if fixed != None:
                if not isinstance(element, basestring) or element != fixed:
                    results.push(INVALID_VALUE)
                    continue
            # Default checking
            if default != None and element == None:
                element = default

            # Match another node
            match = definition.get('match', None)
            nMatch = definition.get('notMatch', None)
            if match != None:
                if self._find_value( match, all_data ) != element:
                    return INVALID_MATCH
            if nMatch != None:
                if self._find_value( nMatch, all_data ) == element:
                    return INVALID_MATCH

            opts = {}
            for option in ('minLength', 'maxLength', 'complexType'):
                opts[option] = definition.get(option, None)

            # Element type checking
            result = self._validate_type( dataType, element, **opts )
            if result:
               results.append(result)

        if len(results) > 0:
            return proped and results[0] or results
        return NO_ERROR


    def _validate_type(self, typeName, data, **opts):
        """Internal method for validating a single data type"""
        definition = self._definition
        oSimpleType = definition['simpleTypes'].get(typeName, None)
        complexType = definition['complexTypes'].get(typeName,
                        opts.get('complexType', None))

        if isinstance(data, bool):
            data = data and 'true' or 'false'

        if complexType:
            if isinstance(data, dict):
                return self._validate_elements( complexType, data )
            else:
                return INVALID_COMPLEX
        elif oSimpleType:
            simpleType = oSimpleType.copy()
            simpleType.update(opts)
            base    = simpleType.get('base',    None)
            pattern = simpleType.get('pattern', None)
            custom  = simpleType.get('custom',  None)

            # Base type check
            if base:
                err = self._validate_type( base, data )
                if err:
                    return err

            # Pattern type check, assumes edge detection
            if pattern:
                try:
                    if not re.match("^%s$" % pattern, str(data)):
                        return INVALID_PATTERN((pattern, data))
                except TypeError:
                    return INVALID_PATTERN((typeName, type(data)))

            # Custom method check
            if custom:
                if not callable(custom):
                    return INVALID_CUSTOM
                failure = custom(data, simpleType)
                if failure:
                    return failure

            # Maximum Length check
            maxLength = simpleType.get('maxLength', None)
            if maxLength != None and len(data) > int(maxLength):
                return INVALID_MAXLENGTH

            # Minimum Length check
            minLength = simpleType.get('minLength', None)
            if minLength != None and len(data) < int(minLength):
                return INVALID_MINLENGTH

            # Check Enumeration
            enum = simpleType.get('enumeration', None)
            if enum:
                if not isinstance(enum, list):
                    raise Exception("Validator Error: Enumberation not a list")
                if not data in enum:
                    return INVALID_ENUMERATION

            # This over-writes the data, so be careful
            try:
                data = long(data)
            except:
                pass

            for testName in ('minInclusive', 'maxInclusive', 'minExclusive',
                         'maxExclusive', 'fractionDigits'):
                operator = simpleType.get(testName, None)
                if operator != None:
                    if not isinstance(data, long):
                        return INVALID_NUMBER
                    if testName == 'minInclusive' and data < operator:
                        return INVALID_MIN_RANGE
                    if testName == 'maxInclusive' and data > operator:
                        return INVALID_MAX_RANGE
                    if testName == 'minExclusive' and data <= operator:
                        return INVALID_MIN_RANGE
                    if testName == 'maxExclusive' and data >= operator:
                        return INVALID_MAX_RANGE
                    # This is a bit of a hack, but I don't care so much.
                    if testName == 'fractionDigits' and '.' in str(data):
                        if len(str(data).split('.')[1]) > operator:
                            return INVALID_FRACTION
        else:
            raise NoTypeFound("Can't find type '%s'" % typeName)
        return NO_ERROR

    def _find_value(self, path, data):
        """Internal method for finding a value match (basic xpath)"""
        # Remove root path, and stop localisation
        if path[0] == '/':
            data  = self.current_root_data
            paths = path[1:].split('/')
        else:
            paths = path.split('/')

        for segment in paths:
            if isinstance(data, dict):
                try:
                    data = data[segment]
                except KeyError:
                    #logging.warn("Validator Error: Can't find key for '%s'-> %s in %s" % (path, segment, str(paths)))
                    return INVALID_XPATH(path)
            else:
                #logging.warn("Validator Error: Can't find value for '%s'-> %s in %s" % (path, segment, str(paths)))
                INVALID_XPATH(path)
        return data

    def _load_file(self, filename, definition=None):
        if not os.path.exists(filename):
            raise IOError("file doesn't exist: %s" % filename)

        with open(filename, 'r') as fhl:
            content = fh.read()
  
            if content[0] == '<':
                # XML File, parse and load
                parser = ParseXML( filename )
                if definition and 'XMLSchema' in content:
                    return parser.definition
                else:
                    return parser.data

        raise IOError("Couldn't open and parse: %s" % filename)
