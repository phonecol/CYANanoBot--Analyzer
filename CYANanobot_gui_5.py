# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import subprocess
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1117, 849)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(420, 60, 310, 33))
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAutoFillBackground(False)
        self.label.setObjectName("label")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(50, 100, 1041, 131))
        self.groupBox.setObjectName("groupBox")
        self.iap_button = QtWidgets.QPushButton(self.groupBox)
        self.iap_button.setGeometry(QtCore.QRect(0, 30, 260, 80))
        self.iap_button.setObjectName("iap_button")
        self.cep_button = QtWidgets.QPushButton(self.groupBox)
        self.cep_button.setEnabled(True)
        self.cep_button.setGeometry(QtCore.QRect(260, 30, 260, 80))
        self.cep_button.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.cep_button.setObjectName("cep_button")
        self.rap_button = QtWidgets.QPushButton(self.groupBox)
        self.rap_button.setGeometry(QtCore.QRect(520, 30, 260, 80))
        self.rap_button.setObjectName("rap_button")
        self.pdp_button = QtWidgets.QPushButton(self.groupBox)
        self.pdp_button.setGeometry(QtCore.QRect(780, 30, 260, 80))
        self.pdp_button.setObjectName("pdp_button")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(110, 740, 131, 16))
        self.label_3.setObjectName("label_3")
        self.cmd_line_run = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.cmd_line_run.setGeometry(QtCore.QRect(110, 660, 931, 71))
        self.cmd_line_run.setObjectName("cmd_line_run")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(430, 240, 321, 341))
        self.groupBox_2.setObjectName("groupBox_2")
        self.formLayoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 40, 301, 171))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.ia_paper_sensor = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.ia_paper_sensor.setObjectName("ia_paper_sensor")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.ia_paper_sensor)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.ia_cn_concentration = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.ia_cn_concentration.setObjectName("ia_cn_concentration")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.ia_cn_concentration)
        self.label_5 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.ia_exposure = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.ia_exposure.setObjectName("ia_exposure")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.ia_exposure)
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.ia_resolution = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.ia_resolution.setObjectName("ia_resolution")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.ia_resolution)
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.ia_time_interval = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.ia_time_interval.setObjectName("ia_time_interval")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.ia_time_interval)
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.ia_directory = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.ia_directory.setObjectName("ia_directory")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.ia_directory)
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.groupBox_2)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(10, 220, 301, 31))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.ia_clear_all_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_6)
        self.ia_clear_all_button.setObjectName("ia_clear_all_button")
        self.horizontalLayout_6.addWidget(self.ia_clear_all_button)
        self.ia_apply_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_6)
        self.ia_apply_button.setObjectName("ia_apply_button")
        self.horizontalLayout_6.addWidget(self.ia_apply_button)
        self.ia_run_button = QtWidgets.QPushButton(self.groupBox_2)
        self.ia_run_button.setGeometry(QtCore.QRect(90, 260, 141, 71))
        self.ia_run_button.setObjectName("ia_run_button")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(430, 240, 321, 241))
        self.groupBox_4.setObjectName("groupBox_4")
        self.formLayoutWidget_5 = QtWidgets.QWidget(self.groupBox_4)
        self.formLayoutWidget_5.setGeometry(QtCore.QRect(10, 20, 301, 101))
        self.formLayoutWidget_5.setObjectName("formLayoutWidget_5")
        self.formLayout_5 = QtWidgets.QFormLayout(self.formLayoutWidget_5)
        self.formLayout_5.setContentsMargins(0, 0, 0, 0)
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_29 = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.label_29.setObjectName("label_29")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_29)
        self.ce_paper_sensor = QtWidgets.QLineEdit(self.formLayoutWidget_5)
        self.ce_paper_sensor.setObjectName("ce_paper_sensor")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.ce_paper_sensor)
        self.label_30 = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.label_30.setObjectName("label_30")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_30)
        self.ce_id = QtWidgets.QLineEdit(self.formLayoutWidget_5)
        self.ce_id.setObjectName("ce_id")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.ce_id)
        self.label_32 = QtWidgets.QLabel(self.formLayoutWidget_5)
        self.label_32.setObjectName("label_32")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_32)
        self.ce_rf2 = QtWidgets.QLineEdit(self.formLayoutWidget_5)
        self.ce_rf2.setObjectName("ce_rf2")
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.ce_rf2)
        self.ce_run_button = QtWidgets.QPushButton(self.groupBox_4)
        self.ce_run_button.setGeometry(QtCore.QRect(90, 160, 141, 71))
        self.ce_run_button.setObjectName("ce_run_button")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_4)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 120, 301, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.ce_clear_all_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.ce_clear_all_button.setObjectName("ce_clear_all_button")
        self.horizontalLayout_3.addWidget(self.ce_clear_all_button)
        self.ce_apply_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.ce_apply_button.setObjectName("ce_apply_button")
        self.horizontalLayout_3.addWidget(self.ce_apply_button)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(430, 240, 321, 361))
        self.groupBox_3.setObjectName("groupBox_3")
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_3)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(10, 40, 301, 191))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.formLayout_3 = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_15 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_15.setObjectName("label_15")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.pd_paper_sensor = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.pd_paper_sensor.setObjectName("pd_paper_sensor")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.pd_paper_sensor)
        self.label_16 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_16.setObjectName("label_16")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.pd_filename = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.pd_filename.setObjectName("pd_filename")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pd_filename)
        self.label_17 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_17.setObjectName("label_17")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.pd_rf1 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.pd_rf1.setObjectName("pd_rf1")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.pd_rf1)
        self.label_18 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_18.setObjectName("label_18")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.pd_rf2 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.pd_rf2.setObjectName("pd_rf2")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.pd_rf2)
        self.label_19 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_19.setObjectName("label_19")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.pd_directory = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.pd_directory.setObjectName("pd_directory")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.pd_directory)
        self.label_20 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_20.setObjectName("label_20")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.pd_subdirectory = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.pd_subdirectory.setObjectName("pd_subdirectory")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.pd_subdirectory)
        self.label_21 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_21.setObjectName("label_21")
        self.formLayout_3.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.pd_num_rows = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.pd_num_rows.setObjectName("pd_num_rows")
        self.formLayout_3.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.pd_num_rows)
        self.pd_run_button = QtWidgets.QPushButton(self.groupBox_3)
        self.pd_run_button.setGeometry(QtCore.QRect(90, 280, 141, 71))
        self.pd_run_button.setObjectName("pd_run_button")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.groupBox_3)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 240, 301, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pd_clear_all_button = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pd_clear_all_button.setObjectName("pd_clear_all_button")
        self.horizontalLayout.addWidget(self.pd_clear_all_button)
        self.pd_apply_button = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pd_apply_button.setObjectName("pd_apply_button")
        self.horizontalLayout.addWidget(self.pd_apply_button)
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setEnabled(True)
        self.groupBox_5.setGeometry(QtCore.QRect(430, 240, 321, 321))
        self.groupBox_5.setObjectName("groupBox_5")
        self.formLayoutWidget_7 = QtWidgets.QWidget(self.groupBox_5)
        self.formLayoutWidget_7.setGeometry(QtCore.QRect(10, 20, 301, 61))
        self.formLayoutWidget_7.setObjectName("formLayoutWidget_7")
        self.formLayout_7 = QtWidgets.QFormLayout(self.formLayoutWidget_7)
        self.formLayout_7.setContentsMargins(0, 0, 0, 0)
        self.formLayout_7.setObjectName("formLayout_7")
        self.label_35 = QtWidgets.QLabel(self.formLayoutWidget_7)
        self.label_35.setObjectName("label_35")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_35)
        self.ra_paper_sensor = QtWidgets.QLineEdit(self.formLayoutWidget_7)
        self.ra_paper_sensor.setObjectName("ra_paper_sensor")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.ra_paper_sensor)
        self.label_36 = QtWidgets.QLabel(self.formLayoutWidget_7)
        self.label_36.setObjectName("label_36")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_36)
        self.ra_id = QtWidgets.QLineEdit(self.formLayoutWidget_7)
        self.ra_id.setObjectName("ra_id")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.ra_id)
        self.ra_run_button = QtWidgets.QPushButton(self.groupBox_5)
        self.ra_run_button.setGeometry(QtCore.QRect(100, 250, 141, 71))
        self.ra_run_button.setObjectName("ra_run_button")
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.groupBox_5)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(10, 210, 301, 31))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.ra_clear_all_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        self.ra_clear_all_button.setObjectName("ra_clear_all_button")
        self.horizontalLayout_5.addWidget(self.ra_clear_all_button)
        self.ra_apply_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        self.ra_apply_button.setObjectName("ra_apply_button")
        self.horizontalLayout_5.addWidget(self.ra_apply_button)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox_5)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 90, 301, 91))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ra_lr_radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.ra_lr_radioButton.setObjectName("ra_lr_radioButton")
        self.verticalLayout.addWidget(self.ra_lr_radioButton)
        self.ra_mlr_radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.ra_mlr_radioButton.setObjectName("ra_mlr_radioButton")
        self.verticalLayout.addWidget(self.ra_mlr_radioButton)
        self.ra_pr_radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.ra_pr_radioButton.setObjectName("ra_pr_radioButton")
        self.verticalLayout.addWidget(self.ra_pr_radioButton)
        self.ra_mpr_radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.ra_mpr_radioButton.setObjectName("ra_mpr_radioButton")
        self.verticalLayout.addWidget(self.ra_mpr_radioButton)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1117, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pd_apply_button.clicked.connect(self.roi_extraction)
        self.ce_apply_button.clicked.connect(self.color_extraction)
        self.ra_apply_button.clicked.connect(self.regression_analysis)
        self.ra_apply_button.clicked.connect(self.regression_analysis)

        self.ra_clear_all_button.clicked.connect(self.clear_fields_ra)
        self.ce_clear_all_button.clicked.connect(self.clear_fields_ce)
        self.pd_clear_all_button.clicked.connect(self.clear_fields_pd)


        self.iap_button.clicked.connect(self.iap_button_clicked)
        self.rap_button.clicked.connect(self.rap_button_clicked)
        self.cep_button.clicked.connect(self.cep_button_clicked)
        self.pdp_button.clicked.connect(self.pdp_button_clicked)


        self.groupBox_2.hide()
        self.groupBox_3.hide()
        self.groupBox_4.hide()
        self.groupBox_5.hide()

        self.ce_run_button.clicked.connect(self.run_cep)
        self.pd_run_button.clicked.connect(self.run_pdp)





    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "CYANanoBot Software V.1.0"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.iap_button.setText(_translate("MainWindow", "Image Acquisition Program"))
        self.cep_button.setText(_translate("MainWindow", "Color Extraction Program"))
        self.rap_button.setText(_translate("MainWindow", "Regression Analysis Program"))
        self.pdp_button.setText(_translate("MainWindow", "Paper Detection Program"))
        self.label_3.setText(_translate("MainWindow", "Command Line"))
        self.cmd_line_run.setPlainText(_translate("MainWindow", "command line edit"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Image Acquisition Program Parameters"))
        self.label_2.setText(_translate("MainWindow", "Name of Paper Sensor "))
        self.label_4.setText(_translate("MainWindow", "Cyanide Concentration"))
        self.label_5.setText(_translate("MainWindow", "Exposure"))
        self.label_6.setText(_translate("MainWindow", "Resolution"))
        self.label_7.setText(_translate("MainWindow", "Time Interval"))
        self.label_8.setText(_translate("MainWindow", "Directory"))
        self.ia_clear_all_button.setText(_translate("MainWindow", "Clear All"))
        self.ia_apply_button.setText(_translate("MainWindow", "Apply"))
        self.ia_run_button.setText(_translate("MainWindow", "Run Program"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Color Extraction Program Parameters"))
        self.label_29.setText(_translate("MainWindow", "Name of Paper Sensor "))
        self.label_30.setText(_translate("MainWindow", "images id"))
        self.label_32.setText(_translate("MainWindow", "ROI 2"))
        self.ce_run_button.setText(_translate("MainWindow", "Run Program"))
        self.ce_clear_all_button.setText(_translate("MainWindow", "Clear All"))
        self.ce_apply_button.setText(_translate("MainWindow", "Apply"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Paper Detection Program Parameters"))
        self.label_15.setText(_translate("MainWindow", "Name of Paper Sensor "))
        self.label_16.setText(_translate("MainWindow", "Filename"))
        self.label_17.setText(_translate("MainWindow", "ROI 1"))
        self.label_18.setText(_translate("MainWindow", "ROI 2"))
        self.label_19.setText(_translate("MainWindow", "Directory"))
        self.label_20.setText(_translate("MainWindow", "SubDirectory"))
        self.label_21.setText(_translate("MainWindow", "Number of Rows"))
        self.pd_run_button.setText(_translate("MainWindow", "Run Program"))
        self.pd_clear_all_button.setText(_translate("MainWindow", "Clear All"))
        self.pd_apply_button.setText(_translate("MainWindow", "Apply"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Regression Analysis Program Parameters"))
        self.label_35.setText(_translate("MainWindow", "Name of Paper Sensor "))
        self.label_36.setText(_translate("MainWindow", "images id"))
        self.ra_run_button.setText(_translate("MainWindow", "Run Program"))
        self.ra_clear_all_button.setText(_translate("MainWindow", "Clear All"))
        self.ra_apply_button.setText(_translate("MainWindow", "Apply"))
        self.ra_lr_radioButton.setText(_translate("MainWindow", "Linear Regression"))
        self.ra_mlr_radioButton.setText(_translate("MainWindow", "Multiple Linear Regression"))
        self.ra_pr_radioButton.setText(_translate("MainWindow", "Polynomial Regressiion"))
        self.ra_mpr_radioButton.setText(_translate("MainWindow", "Multiple Polynomial Regression"))

    def iap_button_clicked(self):

        # self.groupBox_2.hide()
        self.groupBox_3.hide()
        self.groupBox_4.hide()
        self.groupBox_5.hide()
        self.groupBox_2.show()

    def cep_button_clicked(self):

        self.groupBox_2.hide()
        self.groupBox_3.hide()
        # self.groupBox_4.hide()
        self.groupBox_5.hide()
        self.groupBox_4.show()

    def pdp_button_clicked(self):
        self.groupBox_2.hide()
        # self.groupBox_3.hide()
        self.groupBox_4.hide()
        self.groupBox_5.hide()
        self.groupBox_3.show()


    def rap_button_clicked(self):
        self.groupBox_2.hide()
        self.groupBox_3.hide()
        self.groupBox_4.hide()
        self.groupBox_5.show()

        # self.groupBox_5.hide()


    def clear_fields_ra(self):
        self.ra_paper_sensor.clear()
        self.ra_id.clear()

    def clear_fields_ce(self):
        self.ce_paper_sensor.clear()
        self.ce_id.clear()
        self.ce_rf2.clear()

    def clear_fields_pd(self):
        self.pd_paper_sensor.clear()
        self.pd_rf1.clear()
        self.pd_rf2.clear()
        self.pd_filename.clear()
        self.pd_directory.clear()
        self.pd_subdirectory.clear()
        self.pd_num_rows.clear()

    def clear_fields_ia(self):
        self.ia_paper_sensor.clear()
        self.ia_cn_concentration.clear()
        self.ia_directory.clear()
        self.ia_exposure.clear()
        self.ia_resolution.clear()
        self.ia_time_interval.clear()


    def roi_extraction(self):
        self.cmd_line_run.clear()
        print("Paper sensor detection and cropping program")

        pd_cmd_list = ['python',
                    'paper_sensor_detection_program.py',
                    '-fn',
                    self.pd_filename.text(),
                    '-ps',
                    self.pd_paper_sensor.text(),
                    '-rf1',
                    self.pd_rf1.text(),
                    '-rf2',
                    self.pd_rf2.text(),
                    '-cif',
                    self.pd_directory.text(),
                    '-cisf',
                    self.pd_subdirectory.text(),
                    '-nr',
                    self.pd_num_rows.text()]
        print(pd_cmd_list)
        pd_cmd_str = ' '.join(pd_cmd_list)
        print(pd_cmd_str)
        self.cmd_line_run.insertPlainText(pd_cmd_str)
        # pd_cmd_list1= self.cmd_line_run.toPlainText().split(" ")
        # print(pd_cmd_list1)
        # pd_cmd_str1= ' '.join(pd_cmd_list1)
        # print(pd_cmd_str1)
        # subprocess.run(pd_cmd_list1)
        # subprocess.run(['python', 'paper_sensor_detection_program.py', '-fn', 'new_sensor','-id2','10', '-rf1','ROI_newsensor', '-rf2', 'ROI2_newsensor', '-cif','captured_images3','-cisf', 'data_gathering_newsensor', '-nr','3', '-si','False'])


    def image_acquisition(self):
        self.cmd_line_run.clear()
        print("Paper sensor detection and cropping program")

        ia_cmd_list = ['python',
                    'image_acquisition_program.py',
                    '-cn',
                    self.ia_cn_concentration.text(),
                    '-ps',
                    self.ia_paper_sensor.text(),
                    '-in',
                    self.ia_time_interval_slider.text(),
                    '-r',
                    self.ia_resolution.text(),
                    '-cif',
                    self.ia_directory.text(),
                    '-e',
                    self.ia_exposure.text(),
                    '-i',
                    self.ia_num_images_slider.text()
                    ]
        print(ia_cmd_list)
        ia_cmd_str = ' '.join(ia_cmd_list)
        print(ia_cmd_str)
        self.cmd_line_run.insertPlainText(ia_cmd_str)





    def color_extraction(self):
        self.cmd_line_run.clear()
        print("Color Extraction Program")

        ce_cmd_list = ['python',
                    'color_extraction_program.py',
                    '-fn',
                    self.ce_paper_sensor.text(),
                    '-id2',
                    self.ce_id.text(),
                    '-rf2',
                    self.ce_rf2.text()
                    ]
        print(ce_cmd_list)
        ce_cmd_str = ' '.join(ce_cmd_list)
        print(ce_cmd_str)
        self.cmd_line_run.insertPlainText(ce_cmd_str)
        # ce_cmd_list1= self.cmd_line_run.toPlainText().split(" ")
        # print(ce_cmd_list1)
        # ce_cmd_str1= ' '.join(ce_cmd_list1)
        # print(ce_cmd_str1)
        # subprocess.run(ce_cmd_list1)
        # subprocess.run(['python', 'paper_sensor_detection_program.py', '-fn', 'new_sensor','-id2','10', '-rf1','ROI_newsensor', '-rf2', 'ROI2_newsensor', '-cif','captured_images3','-cisf', 'data_gathering_newsensor', '-nr','3', '-si','False'])

    def regression_analysis(self):
        self.cmd_line_run.clear()
        print("Regression Analysis Program")

        ra_cmd_list = ['python',
                    'create_regression_model.py',
                    '-fn',
                    self.ra_paper_sensor.text(),
                    '-id2',
                    self.ra_id.text(),
                    ]
        print(ra_cmd_list)
        ra_cmd_str = ' '.join(ra_cmd_list)
        print(ra_cmd_str)
        self.cmd_line_run.insertPlainText(ra_cmd_str)
            # ra_cmd_list1= self.cmd_line_run.toPlainText().split(" ")
            # print(ra_cmd_list1)
            # ra_cmd_str1= ' '.join(ra_cmd_list1)
            # print(ra_cmd_str1)
            # subprocess.run(ra_cmd_list1)

    def run_iap(self):
        # cmd_line = self.cmd_line_run.text()
        iap_cmd_list1= self.cmd_line_run.toPlainText().split(" ")
        # print(cmd_line)
        print(iap_cmd_list1)
        subprocess.run(iap_cmd_list1)

    def run_cep(self):
        # cmd_line = self.cmd_line_run.text()
        cep_cmd_list1= self.cmd_line_run.toPlainText().split(" ")
        # print(cmd_line)
        print(cep_cmd_list1)
        subprocess.run(cep_cmd_list1)
        # pd_cmd_str1= ' '.join(pd_cmd_list1)
        # print(pd_cmd_str1)
    def run_pdp(self):
        # cmd_line = self.cmd_line_run.text()
        pdp_cmd_list1= self.cmd_line_run.toPlainText().split(" ")
        # print(cmd_line)
        print(pdp_cmd_list1)
        subprocess.run(pdp_cmd_list1)

    def run_rap(self):
        # cmd_line = self.cmd_line_run.text()
        rap_cmd_list1= self.cmd_line_run.toPlainText().split(" ")
        # print(cmd_line)
        print(rap_cmd_list1)
        subprocess.run(rap_cmd_list1)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
