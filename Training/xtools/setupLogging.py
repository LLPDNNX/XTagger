import logging
import sys

def setupLogging(level=logging.DEBUG):
    formatterOutput = logging.Formatter("[%(levelname)s] %(message)s")
    formatterError = logging.Formatter("[%(levelname)s] %(message)s [%(filename)s:%(lineno)d]")
    streamOutputHandler = logging.StreamHandler(stream=sys.stdout)
    streamOutputHandler.setLevel(level)
    streamErrorHandler = logging.StreamHandler(stream=sys.stderr)
    streamErrorHandler.setLevel(logging.ERROR)
    streamOutputHandler.setFormatter(formatterOutput)
    streamErrorHandler.setFormatter(formatterError)
    rootLogger = logging.getLogger()
    rootLogger.setLevel(level)
    rootLogger.addHandler(streamOutputHandler)
    rootLogger.addHandler(streamErrorHandler)


