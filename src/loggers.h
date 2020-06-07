#pragma once

#include <boost/log/common.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/utility/setup/from_stream.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>
#include <boost/log/trivial.hpp>

namespace logging
{
    using severity_t = boost::log::trivial::severity_level;
    using logger_t = boost::log::sources::severity_logger<severity_t>;

    BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(logger_main, logger_t)
}
