import logging
import logging.config

import src.utility.constants as CONST

# Logging
logging.config.fileConfig(CONST.log_config)
logger = logging.getLogger(__name__)