#include "util.h"

void outfmt::validate(boost::any & v, const std::vector<std::string> & values, outfmt::OutputFormat * target_type, int)
{
    using namespace boost::program_options;

    // Make sure no previous assignment to 'a' was made.
    validators::check_first_occurrence(v);
    // Extract the first string from 'values'. If there is more than
    // one string, it's an error, and exception will be thrown.
    const std::string & s = validators::get_single_string(values);

    try
    {
        v = boost::any(formats.at(s));
    }
    catch (const std::out_of_range &)
    {
        throw validation_error(validation_error::invalid_option_value);
    }
}