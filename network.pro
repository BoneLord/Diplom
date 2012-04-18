#-------------------------------------------------
#
# Project created by QtCreator 2012-02-24T13:18:55
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = network
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    neuralnetwork.cpp \
    layer.cpp \
    trainingsetelement.cpp

HEADERS += \
    neuralnetwork.h \
    layer.h \
    trainingsetelement.h
