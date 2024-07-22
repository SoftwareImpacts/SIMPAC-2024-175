#=================================================================================
#!/usr/bin/python
# encoding: UTF-8
#
# FILE: caldintav.py
#
#
# DESCRIPTION:
#
# OPTIONS:
# REQUIREMENTS:
# AUTHOR: Khanh Nguyen Gia, khanh@mecanica.upm.es or khanhng1982@yahoo.com
# WEB: http://w3.mecanica.upm.es/~khanh
# VERSION: 1.0.10
# CREATED: 05-09-2017
# LICENSE: GNU AFFERO GENERAL PUBLIC LICENSE
#=================================================================================

from .functions import *

import sys

from PyQt5 import QtCore, QtGui, QtWidgets

import caldintav as cdt
from os import path
#------------------------------------------------------------------------
direction = path.abspath(path.dirname(cdt.__file__))
#------------------------------------------------------------------------

from .caldintav_designer import Ui_MainWindow

import pickle 

import csv

from os.path import isdir

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

from matplotlib.backends.backend_qt4agg import  NavigationToolbar2QT as NavigationToolbar

from matplotlib import rc

from pprint import pprint

#-----------------------------------------------------------------------------------
# ============= Text rendering latex ====================================
font = {'size':10}
rc('font',**font)
rc('lines',linewidth=2.0,markersize=4)
rc('legend',fontsize=8,borderaxespad=0.5,fancybox=True, shadow=False)
rc('axes',linewidth=0.5)
rc('xtick',labelsize=10)
rc('ytick',labelsize=10)
rc('grid',linewidth=0.5)
# =======================================================================

class caldintav(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        QtWidgets.QWidget.__init__(self,parent)
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        # set image upm and caminos 
        self.ui.label.setPixmap(QtGui.QPixmap(direction+"/upm.png"))
        self.ui.label_2.setPixmap(QtGui.QPixmap(direction+"/caminos.png"))
        # -------
        # File in Menubar
        # connect signal to the new project
        self.ui.new_project.triggered.connect(self.new_project_dialog)
        # connect signal to the open project
        self.ui.open_project.triggered.connect(self.open_project_dialog)
        # connect signal to the save project
        self.ui.save_project.triggered.connect(self.save_project_dialog)
        # coonect signal to action quit
        self.ui.actionQuit.triggered.connect(self.closeApp)
        # -------
        self.ui.project_name.editingFinished.connect(self.create_project_folder)
        # -------
        # Help in Menubar
        # connect signal to the About
        self.about = AboutDialog(self)
        self.ui.actionAbout.triggered.connect(self.open_about)
        self.about.license.clicked.connect(self.open_license)
        # connect signal to the User guide
        self.ui.actionUser_Guide.triggered.connect(self.open_manual)
        # -------
        # Bridge data ------
        self.BridgeData = cell_struct()
        # Initial data
        if self.ui.bridge_types.currentIndex() == 0:
            self.ui.BridgeProperties.setRowCount(0)
            self.ui.number_span.setMinimum(0)
            self.ui.number_span.setValue(0)
        self.ui.bridge_types.currentIndexChanged.connect(self.adjust_bridge_data)
        # connect signal to save bridge button
        self.ui.save_bridge.clicked.connect(self.save_bridge)
        # connect signal to clear bridge button
        self.ui.clear_bridge.clicked.connect(self.clear_bridge)
        # connect signal to export bridge data to csv file button
        self.ui.export_bridge.clicked.connect(self.export_bridge_csv)
        # connect signal to import bridge data from csv file button
        self.ui.import_bridge.clicked.connect(self.import_bridge_csv)
        # -------
        # Train data ------
        #
        # create list store for the available trains
        list_trains = QtGui.QStandardItemModel()
        [list_trains.appendRow(QtGui.QStandardItem('HSLM-A'+str(i))) for i in range(1,11)]
        self.ui.list_trains.setModel(list_trains)
        self.ui.list_trains.clicked.connect(self.get_selected_item_list)
        # add trains to the list of selected trains
        self.list_selected_trains = QtGui.QStandardItemModel()
        self.ui.list_selected.setModel(self.list_selected_trains)
        self.ui.list_selected.clicked.connect(self.get_selected_item_to_remove)
        # connect signal to button "Select an item"
        self.ui.selectButton.clicked.connect(self.selectButton_click)
        # connect signal to button "Remove an item"
        self.ui.removeButton.clicked.connect(self.removeButton_click)
        # connect signal to button "Select all items"
        self.ui.selectAll.clicked.connect(self.selectAll)
        # connect signal to button "Remove all items"
        self.ui.removeAll.clicked.connect(self.removeAll)
        # connect signal to button "Delete train"
        self.ui.delete_train.clicked.connect(self.delete_train)
        # connect signal to button "Show train data"
        self.ui.show_train.clicked.connect(self.show_train)
        # New train 
        self.add_train = []
        # new train from table
        self.new_train_table = NewTrainDialog(self)
        # connect signal to save new train from train data
        self.ui.save_train.clicked.connect(self.save_train)
        self.new_train_table.saveButton.clicked.connect(self.saveNewTrain)
        # connect signal to button "clear train data"
        self.ui.clear_table.clicked.connect(self.clear_table)
        #
        self.ui.number_axles.valueChanged.connect(self.adjust_train_table)
        self.ui.number_axles.setValue(0)
        # connect signal to button "Load train's file"
        self.ui.load_train.clicked.connect(self.load_train_file)
        self.add_train_dialog = AddTrainDialog(self)
        self.add_train_dialog.saveButton.clicked.connect(self.saveAddTrain)
        # new train for interaction dynamic options
        self.newTrainInter = InteractionTrainDialog(self)
        self.ui.add_data.clicked.connect(self.saveTrainInter)
        # connect signal for the save button in the new train for interaction dialog
        self.newTrainInter.saveButton.clicked.connect(self.saveNewTrainInter)
        self.newTrainInter.import_csv_file.clicked.connect(self.opencsvfile)
        self.newTrainInter.clearData.clicked.connect(self.clearTrainInteData)
        self.TrainInteList = []
        self.add_trainInte = []
        # connect signal for the "submit selected trains" button
        self.ui.submit_train.clicked.connect(self.submitTrains)
        # connect signal for the "Delete selected trains" button
        self.ui.delete_selected_train.clicked.connect(self.deleteSubmittedTrains)
        # -------
        # Analysis Options ------
        self.ui.numberCores.setMaximum(1)
        self.ui.MPdistance.setMaximum(0.)
        self.ui.sleeper_separation.setMaximum(0.60)
        # connect signal for the "Save options" button
        self.ui.AOsaveoptions.clicked.connect(self.saveoptions)
        # connect signal for the "Clear options" button
        self.ui.AOclearoptions.clicked.connect(self.clearoptions)
        # connect signal when the load distribution checkbox is ckecked
        self.ui.loadDist.toggled.connect(self.setSleeperSeperation)
        # connect signal when the parallel button is ckecked 
        self.ui.changeMonitoredPoint.toggled.connect(self.setDistance)
        # connect signal when the parallel button is ckecked 
        self.ui.ParallelComp.toggled.connect(self.setNumberCores)
        # coonect signal to warning message when interaction analysis is selected
        self.ui.interaction.toggled.connect(self.messageInter)
        # connect signal to warning message when DER analysis is selected
        self.ui.DER.toggled.connect(self.messageDER)
        # connect signal to warning message when LIR analysis is selected
        self.ui.LIR.toggled.connect(self.messageLIR)
        # Submit Job ------------
        # connect signal for the submit job button
        self.ui.submitJob.clicked.connect(self.submitjob)
        # Global results --------
        # connect selected trains table in tab "global results" with the selected trains in Train Data
        self.ui.listTrainsResults.clicked.connect(self.get_selected_item_listTrainsResults)
        self.plot_selected_trains = QtGui.QStandardItemModel()
        self.ui.plotTrains.setModel(self.plot_selected_trains)
        self.ui.plotTrains.clicked.connect(self.get_selected_item_to_remove)
        # connect signal for the include trains button
        self.ui.includeTrain.clicked.connect(self.IncludeTrain_click)
        # connect signal for the remove train button
        self.ui.removeTrainsResults.clicked.connect(self.removeTrain_click)
        # Assign plot area for plotting global results
        self.plotGlobalResults = QtWidgets.QVBoxLayout(self.ui.plotArea1)
        self.plotGR = MyGlobalResultsCanvas(self.ui.plotArea1)
        self.plotGlobalResults.addWidget(self.plotGR)
        #
        self.ui.label_70.setAutoFillBackground(True)
        self.ui.label_70.setStyleSheet("QLabel { background-color: white; color: red; }")     
        self.ui.label_75.setAutoFillBackground(True)
        self.ui.label_75.setStyleSheet("QLabel { background-color: white; color: red; }")     
        self.ui.label_73.setAutoFillBackground(True)
        self.ui.label_73.setStyleSheet("QLabel { background-color: white; color: red; }")     
        self.ui.label_77.setAutoFillBackground(True)
        self.ui.label_77.setStyleSheet("QLabel { background-color: white; color: red; }")
        # connect signal for plot results button
        self.ui.plotResults.clicked.connect(self.plotResults)
        # connect signal for save figure button
        self.ui.savePlot.clicked.connect(self.saveFigures)
        # connect signal for Max. Accel. Limit button
        self.ui.plotAccelLimit.clicked.connect(self.plotMaxAccelLimit)
        # connect signal for plot UIC limit
        self.ui.plotUIC.clicked.connect(self.plotUIC)
        #---------------------------
        #----- History Results -----
        #---------------------------
        self.ui.comboSelectedTrains.addItem("")
        self.ui.comboSelectedTrains.setItemText(0,"Select an item")
        self.ui.comboVelo.addItem("")
        self.ui.comboVelo.setItemText(0,"Select an item")
        # Assign plot area for history results
        self.plotHistoryResults = QtWidgets.QVBoxLayout(self.ui.plotArea2)
        self.plotHR = MyHistoryResultsCanvas(self.ui.plotArea2)
        self.navi_toolbar = NavigationToolbar(self.plotHR,self) 
        self.plotHistoryResults.addWidget(self.plotHR)
        self.plotHistoryResults.addWidget(self.navi_toolbar)
        # connect signal for plot accel. button
        self.ui.plotAccelTF.clicked.connect(self.plotAccelTF)
        # connect signal for plot disp. button
        self.ui.plotDispTF.clicked.connect(self.plotDispTF)
        # connect signal for clear figure button
        self.ui.clearFigure.clicked.connect(self.clearFigure)
        #----------------------------
        #----- Modal Shapes ---------
        #----------------------------
        # connect signal for plot vertical mode shape button
        self.ui.plotVModeShape.clicked.connect(self.plotVModeShape)
        # connect signal for plot torsioanl mode shape button
        self.ui.plotTModeShape.clicked.connect(self.plotTModeShape)
        # Assign plot area for mode shape
        self.plotModalModes = QtWidgets.QVBoxLayout(self.ui.plotArea3)
        self.plotMM = MyModalModesCanvas(self.ui.plotArea3)
        self.plotModalModes.addWidget(self.plotMM)
        # connect signal for save figure button
        self.ui.saveModeShape.clicked.connect(self.saveModeFigures)
        # connect signal for clear figure button
        self.ui.clearModeShape.clicked.connect(self.clearModeShape)
        # clipboard
        self.clip = QtWidgets.QApplication.clipboard()
    #----------------------------------------------------------------------
    def keyPressEvent(self,event):
        if (event.modifiers() & QtCore.Qt.ControlModifier):
            selected = self.ui.TrainTable.selectedRanges()
            if  event.key() == QtCore.Qt.Key_C:  # copy
                s = ""
                for r in range(selected[0].topRow(),selected[0].bottomRow()+1):
                    for c in range(selected[0].leftColumn(),selected[0].rightColumn()+1):
                        try:
                            s += str(self.ui.TrainTable.item(r,c).text()) + "\t"
                        except AttributeError:
                            s += "\t"
                    s = s[:-1] + "\n"
                self.clip.setText(s)
            if  event.key() == QtCore.Qt.Key_V: # paste
                try:
                    text = QtWidgets.QApplication.instance().clipboard().text()
                    clip_text = text.splitlines()
                    self.ui.TrainTable.setRowCount(len(clip_text))
                    self.ui.number_axles.setValue(len(clip_text))
                    for r in range(len(clip_text)):
                        row = clip_text[r].split()
                        for c in range(len(row)):
                            self.ui.TrainTable.setItem(r,c, QtWidgets.QTableWidgetItem(row[c]))
                except AttributeError:
                    QtWidgets.QMessageBox.question(self,'Error','Can not paste the data', QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            #  
            super(caldintav,self).keyPressEvent(event)
    #----------------------------------------------------------------------
    def new_project_dialog(self):
        text, ok = QtWidgets.QInputDialog.getText(self,'New project','Enter name of project')
        if ok:
            self.project_name = str(text)
            self.ui.project_name.setText(self.project_name)
            self.clear_all()
            return self
    #
    def open_project_dialog(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self,filter=' All files (*);;All project files (*.pkl);;Python files (*.py)', initialFilter='All project files (*.pkl)')
        #
        if  filename[0]:
            fid = open(filename[0], 'rb')
            self.ui.label_35.setText('LOADED')
            # name of project
            self.project_name = pickle.load(fid)
            self.ui.project_name.setText(self.project_name)
            # bridge data
            self.BridgeData = pickle.load(fid)
            if  self.BridgeData.nv == 1:
                self.ui.bridge_types.setCurrentIndex(1)
                self.ui.BC.setCurrentIndex(self.BridgeData.boun+1)
                self.ui.damping_value.setText(str(self.BridgeData.xi*100))
                self.ui.skew_value.setText(str(self.BridgeData.alpha))
                self.ui.number_span.setMaximum(self.BridgeData.nv)
                self.ui.number_span.setMinimum(self.BridgeData.nv)
                self.ui.number_span.setValue(self.BridgeData.nv)
                self.ui.BridgeProperties.setItem(0,0,QtWidgets.QTableWidgetItem(str(self.BridgeData.L)))
                self.ui.BridgeProperties.setItem(0,1,QtWidgets.QTableWidgetItem(str('%.2g' % self.BridgeData.EI)))
                self.ui.BridgeProperties.setItem(0,2,QtWidgets.QTableWidgetItem(str('%.2g' % self.BridgeData.GJ))) 
                self.ui.BridgeProperties.setItem(0,3,QtWidgets.QTableWidgetItem(str(self.BridgeData.m)))
                self.ui.BridgeProperties.setItem(0,4,QtWidgets.QTableWidgetItem(str(self.BridgeData.r)))
                if self.BridgeData.boun == 0:
                    self.ui.BC_summary.setText('Pinned')
                else:
                    self.ui.BC_summary.setText('Fixed')
                self.ui.span_summary.setText(str(self.BridgeData.nv))
                if self.BridgeData.alpha > 0.:
                    self.ui.skew_summary.setText('Yes')
                else:
                    self.ui.skew_summary.setText('No')
                if  self.BridgeData.x != self.BridgeData.L/2.:
                    state = True
                else:
                    state = False
                #
                self.ui.Bstatus_summary.setText('COMPLETED')
            else:
                if self.BridgeData.portico == 1:
                    self.ui.bridge_types.setCurrentIndex(2)
                    if self.BridgeData.x != self.BridgeData.L[0]+self.BridgeData.L[1]/2.:
                        state = True
                    else:
                        state = False
                else:
                    self.ui.bridge_types.setCurrentIndex(3)
                    if self.BridgeData.x != self.BridgeData.L[0]/2.:
                        state = True
                    else:
                        state = False
                self.ui.BC.setCurrentIndex(self.BridgeData.boun+1)
                self.ui.damping_value.setText(str(self.BridgeData.xi*100))
                self.ui.skew_value.setText(str(self.BridgeData.alpha))
                self.ui.number_span.setMaximum(self.BridgeData.nv)
                self.ui.number_span.setMinimum(self.BridgeData.nv)
                self.ui.number_span.setValue(self.BridgeData.nv)
                for i in range(self.BridgeData.nv):
                    self.ui.BridgeProperties.setItem(i,0,QtWidgets.QTableWidgetItem(str(self.BridgeData.L[i])))
                    self.ui.BridgeProperties.setItem(i,1,QtWidgets.QTableWidgetItem(str('%.2g' % self.BridgeData.EI[i])))
                    self.ui.BridgeProperties.setItem(i,2,QtWidgets.QTableWidgetItem(str('%.2g' % self.BridgeData.GJ[i]))) 
                    self.ui.BridgeProperties.setItem(i,3,QtWidgets.QTableWidgetItem(str(self.BridgeData.m[i])))
                    self.ui.BridgeProperties.setItem(i,4,QtWidgets.QTableWidgetItem(str(self.BridgeData.r[i])))
                if self.BridgeData.boun == 0:
                    self.ui.BC_summary.setText('Pinned')
                else:
                    self.ui.BC_summary.setText('Fixed')
                self.ui.span_summary.setText(str(self.BridgeData.nv))
                if self.BridgeData.alpha > 0.:
                    self.ui.skew_summary.setText('Yes')
                else:
                    self.ui.skew_summary.setText('No')
                self.ui.Bstatus_summary.setText('COMPLETED')
            # Train data
            self.add_train = pickle.load(fid)
            self.ui.list_trains.model().clear()
            [self.ui.list_trains.model().appendRow(QtGui.QStandardItem('HSLM-A'+str(i))) for i in range(1,11)]
            if len(self.add_train) != 0:
                for name in self.add_train:
                    self.ui.list_trains.model().appendRow(QtGui.QStandardItem(name[0]))
            self.TrainNames = pickle.load(fid)
            self.list_selected_trains.removeRows(0,self.list_selected_trains.rowCount())
            for i in range(len(self.TrainNames)):
                self.list_selected_trains.appendRow(QtGui.QStandardItem(self.TrainNames[i]))
            self.ui.TotalTrainsSum.setText(str(len(self.TrainNames)))
            self.ui.Tstatus.setText('COMPLETED')
            list_selected_trains = QtGui.QStandardItemModel()
            self.ui.listTrainsResults.setModel(list_selected_trains)
            self.ui.comboSelectedTrains.clear()  # clear all items
            self.ui.comboSelectedTrains.addItem("")
            self.ui.comboSelectedTrains.setItemText(0,'Select an item')
            for i in range(len(self.TrainNames)):
                list_selected_trains.appendRow(QtGui.QStandardItem(self.TrainNames[i]))
                self.ui.comboSelectedTrains.addItem("")
                self.ui.comboSelectedTrains.setItemText(i+1,self.TrainNames[i])
            self.ui.plotTrains.model().clear()
            # Analysis options
            self.AnalysisOptions = pickle.load(fid)
            self.dt = self.AnalysisOptions.dt; self.vmax = self.AnalysisOptions.vmax
            self.vmin = self.AnalysisOptions.vmin; self.incr = self.AnalysisOptions.incr
            self.tra = self.AnalysisOptions.tra; self.type_analysis = self.AnalysisOptions.type_analysis; self.nc = self.AnalysisOptions.nc
            nmod = self.AnalysisOptions.nmod
            self.velo = cell_struct()
            self.velo.ini = self.vmin
            self.velo.final = self.vmax
            self.velo.inc = self.incr; velo = np.arange(self.vmin,self.vmax+self.incr,self.incr)
            self.ui.comboVelo.clear()
            self.ui.comboVelo.addItem("")
            self.ui.comboVelo.setItemText(0, "Select an item")
            for i in range(len(velo)):
                self.ui.comboVelo.addItem("")
                self.ui.comboVelo.setItemText(i+1, str('%.1f' % (velo[i])))
            if self.type_analysis == 1:
                self.ui.movingloads.setChecked(True)
            elif self.type_analysis ==2:
                self.ui.interaction.setChecked(True)
                self.add_trainInte = self.add_train
                self.TrainInteList = []
                for k in self.add_trainInte:
                    self.TrainInteList.append(k[0])
            elif self.type_analysis == 3:
                self.ui.DER.setChecked(True)
            elif self.type_analysis == 4:
                self.ui.LIR.setChecked(True)
            self.ui.Astatus.setText('COMPLETED')
            self.ui.TimeStepSum.setText(str(self.dt) + ' s')
            self.ui.timeIncr.setValue(self.dt)
            self.ui.NumModesSum.setText(str(nmod))
            self.ui.numberModes.setValue(nmod)
            self.ui.spinModeNumber.setMinimum(1)
            self.ui.spinModeNumber.setMaximum(nmod)
            self.ui.MinSpeedSum.setText(str(self.vmin) + ' km/h')
            self.ui.MaxSpeedSum.setText(str(self.vmax) + ' km/h')
            self.ui.SpeedIncrSum.setText(str(self.incr) + ' km/h')
            self.ui.minspeed.setText(str(self.vmin))
            self.ui.maxspeed.setText(str(self.vmax))
            self.ui.speedincr.setText(str(self.incr))
            if  self.tra != 0:
                self.ui.loadDist.setChecked(True)
                self.ui.sleeper_separation.setValue(self.tra)
            else:
                self.ui.loadDist.setChecked(False)
            #
            if self.nc > 1:
                self.ui.ParallelComp.setChecked(True)
                self.ui.numberCores.setValue(self.nc)
            else:
                self.ui.ParallelComp.setChecked(False)
            if state:
                self.ui.MPdistance.setValue(self.BridgeData.x)
                self.ui.changeMonitoredPoint.setChecked(True)
            else:
                self.ui.changeMonitoredPoint.setChecked(False)
            #  check if the project data was recorded with dynamic results
            try:
                w0 = self.BridgeData.wn[0]
            except:
                self.warning_message('The loaded project was saved without the dynamic results. Therefore, you can not consult the dynamic results in Global Results, History results and Modal modes Tabs. \n\n Please submit job to obtain dynamic results !')
                return 0
            # clear global plot
            self.plotGR.axes1.cla()
            self.plotGR.axes2.cla()
            self.plotGR.draw()
            self.ui.label_70.setText("")
            self.ui.label_75.setText("")
            self.ui.label_73.setText("")
            self.ui.label_77.setText("")
            # clear history plot
            self.plotHR.axes1.cla()
            self.plotHR.axes2.cla()
            self.plotHR.draw()
            # clear modes plot
            self.plotMM.axes.cla()
            self.plotMM.draw()
            self.ui.FreqValue.setText("")
            return 
    #
    def save_project_dialog(self):
        if self.ui.label_35.text() == 'COMPLETED':
            filename = QtWidgets.QFileDialog.getSaveFileName(self,filter=' All files (*);;All project files (*.pkl);;Python files (*.py)', initialFilter='All project files (*.pkl)')
            if str(filename[0]).endswith('.pkl'):
                name = filename[0]
            else:
                name = filename[0]+'.pkl' 
            with open(name,'wb') as output:
                pickle.dump(self.project_name,output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.BridgeData, output, pickle.HIGHEST_PROTOCOL)
                if self.type_analysis == 2:
                    pickle.dump(self.add_trainInte, output, pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(self.add_train, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.TrainNames, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.AnalysisOptions, output, pickle.HIGHEST_PROTOCOL)
        else:
            warning = QtWidgets.QMessageBox.warning(self, "WARNING","The dynamic calculation has not yet been performed. If you want to try saving the project, only input project data will be recorded.",QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)
            if warning == QtWidgets.QMessageBox.Yes:
                filename = QtWidgets.QFileDialog.getSaveFileName(self,filter=' All files (*);;All project files (*.pkl);;Python files (*.py)', initialFilter='All project files (*.pkl)')
                if filename[0]:
                    if str(filename[0]).endswith('.pkl'):
                        name = filename[0]
                    else:
                        name = filename[0]+'.pkl' 
                        with open(name,'wb') as output:
                            if len(self.project_name) != 0:
                                pickle.dump(self.project_name,output, pickle.HIGHEST_PROTOCOL)
                            else:
                                self.warning_message('Project name is not defined.')
                            if self.ui.Bstatus_summary.text() == "COMPLETED":
                                pickle.dump(self.BridgeData, output, pickle.HIGHEST_PROTOCOL)
                            else:
                                self.warning_message('Bridge data is not completed.')
                            if self.ui.Tstatus.text() == "COMPLETED":
                                if self.type_analysis ==2:
                                    pickle.dump(self.add_trainInte, output, pickle.HIGHEST_PROTOCOL)
                                else:
                                    pickle.dump(self.add_train, output, pickle.HIGHEST_PROTOCOL)
                                pickle.dump(self.TrainNames, output, pickle.HIGHEST_PROTOCOL)
                            else:
                                self.warning_message('Train data is not completed.')
                            if self.ui.Astatus.text() == "COMPLETED":
                                pickle.dump(self.AnalysisOptions, output, pickle.HIGHEST_PROTOCOL)
                            else:
                                self.warning_message('Analysis options are not completed.')
        return 
    # -------------------------- create project folder -----------------------
    def create_project_folder(self):
        aux = str(self.ui.project_name.text());
        self.project_name = "_".join(aux.split())
        path = getcwd()
        folder = path + '/' + self.project_name
        if  isdir(folder) is False:
            system('mkdir ' + folder)
    #--------------------------- bridge data definition ----------------------
    def adjust_bridge_data(self):
        if  self.ui.bridge_types.currentIndex() ==0:
            self.ui.number_span.setValue(0)
            self.ui.BridgeProperties.setRowCount(0)
        elif self.ui.bridge_types.currentIndex() ==1:
            self.ui.number_span.setValue(1)
            self.ui.number_span.setMinimum(1)
            self.ui.number_span.setMaximum(1)
            self.ui.BridgeProperties.setRowCount(1)
        elif self.ui.bridge_types.currentIndex() ==2:
            self.ui.number_span.setMinimum(3)
            self.ui.number_span.setMaximum(3)
            self.ui.BridgeProperties.setRowCount(3)
            self.ui.number_span.valueChanged.connect(self.adjust_bridge_table)
        elif self.ui.bridge_types.currentIndex() ==3:
            self.ui.number_span.setMinimum(2)
            self.ui.number_span.setMaximum(10)
            self.ui.BridgeProperties.setRowCount(2)
            self.ui.number_span.valueChanged.connect(self.adjust_bridge_table)
    def adjust_bridge_table(self):
        self.ui.BridgeProperties.setRowCount(self.ui.number_span.value())
    def save_bridge(self):
        # Check the bridge type
        if self.ui.bridge_types.currentIndex() == 0:
            self.open_message('The bridge\'s type was not defined. Please select a bridge type and add bridge data!')
            return 0
        else:
            self.ui.span_summary.setText(str(self.ui.number_span.value()))
        # Check boundary conditions
        if  self.ui.BC.currentIndex()==0:
            self.open_message('The bridge\'s boundary conditions were not defined. Please select a boundary condition!')
            return 0
        else:
            if self.ui.BC.currentIndex()==1:
                self.ui.BC_summary.setText('Pinned')
            elif self.ui.BC.currentIndex()==2:
                self.ui.BC_summary.setText('Fixed')
        # Check damping value
        if  self.ui.damping_value.text() == 'Enter value':
            self.open_message('The damping value was not defined. Please provide a value!')
            return 0
        try:
            xi = float(self.ui.damping_value.text())/100.
        except:
            self.open_message('The damping value must be a real value. Please provide a real value!')
            return 0
        # Check skewness
        if  self.ui.skew_value.text() == 'Enter value':
            self.open_message('The skew angle was not defined. Please provide a value!')
            return 0
        else:
            try:
                if  float(self.ui.skew_value.text()) == 0:
                    self.ui.skew_summary.setText('No')
                else:
                    self.ui.skew_summary.setText('Yes')
            except:
                self.open_message('The skew angle must be a real value. Please provide a real value!')
                return 0
        # Obtain the bridge properties
        L,EI,GJ,m,r = [], [], [], [], []
        try :
            for i in range(self.ui.number_span.value()):
                if len(self.ui.BridgeProperties.item(i,0).text()) != 0:
                    L.append(float(self.ui.BridgeProperties.item(i,0).text()))
                if len(self.ui.BridgeProperties.item(i,1).text()) != 0:
                    EI.append(float(self.ui.BridgeProperties.item(i,1).text()))
                if len(self.ui.BridgeProperties.item(i,2).text()) != 0:
                    GJ.append(float(self.ui.BridgeProperties.item(i,2).text()))
                if len(self.ui.BridgeProperties.item(i,3).text()) != 0:
                    m.append(float(self.ui.BridgeProperties.item(i,3).text()))
                if len(self.ui.BridgeProperties.item(i,4).text()) != 0:
                    r.append(float(self.ui.BridgeProperties.item(i,4).text()))
            if  self.ui.number_span.value() == 1:
                self.BridgeData.EI = EI[0]; 
                if  float(self.ui.skew_value.text()) == 0.:
                    self.BridgeData.GJ = EI[0]
                    self.BridgeData.r = np.sqrt(EI[0]*2500./m[0]/3.2e10)
                else:
                    self.BridgeData.GJ = GJ[0]; 
                    self.BridgeData.r = r[0]
                self.BridgeData.L=L[0];
                self.BridgeData.m = m[0]; self.BridgeData.alpha = float(self.ui.skew_value.text()); 
                self.BridgeData.nv =1; self.BridgeData.simply=0; self.BridgeData.x = self.BridgeData.L/2.
                self.BridgeData.xi = xi; self.BridgeData.boun = self.ui.BC.currentIndex() -1; 
                # print(self.BridgeData.boun)
            else:
                self.BridgeData.EI = np.array(EI);
                if  float(self.ui.skew_value.text()) == 0.:
                    self.BridgeData.GJ = np.array(EI); 
                    self.BridgeData.r = np.sqrt(np.array(EI)*2500./np.array(m)/3.2e10)
                else:
                    self.BridgeData.GJ = np.array(GJ)
                    self.BridgeData.r = np.array(r)
                self.BridgeData.L=np.array(L);
                self.BridgeData.m = np.array(m); self.BridgeData.alpha = float(self.ui.skew_value.text());
                self.BridgeData.nv =self.ui.number_span.value(); self.BridgeData.simply=0; 
                self.BridgeData.xi = xi; self.BridgeData.boun = self.ui.BC.currentIndex() -1; 
                if self.ui.bridge_types.currentIndex() == 2:
                    self.BridgeData.portico = 1
                    self.BridgeData.x = self.BridgeData.L[0] + self.BridgeData.L[1]/2.
                else:
                    self.BridgeData.portico = 0
                    self.BridgeData.x = self.BridgeData.L[0]/2.
                #
            self.ui.Bstatus_summary.setText('COMPLETED')
        except:
            self.open_message('Bridge properties were not defined. Please add bridge\'s data!')
            self.BridgeData = cell_struct() 
            self.ui.Bstatus_summary.setText('UNCOMPLETED')
            return
    #signal for clear bridge
    def clear_bridge(self):
        self.ui.bridge_types.setCurrentIndex(0)
        self.ui.BC.setCurrentIndex(0)
        self.ui.damping_value.setText('Enter value')
        self.ui.skew_value.setText('Enter value')
        self.ui.number_span.setMinimum(0)
        self.ui.number_span.setValue(0)
        self.ui.span_summary.setText('')
        self.ui.BC_summary.setText('')
        self.ui.skew_summary.setText('')
        self.ui.Bstatus_summary.setText('DELETED')
        return
    # signal for export bridge data to csv file
    def export_bridge_csv(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self,'Export bridge properties to csv file',filter=' All files (*);;All csv files (*.csv)', initialFilter='All csv files (*.csv)')
        if  filename[0].endswith('.csv'):
            name = filename[0]
        else:
            name = filename[0]+'.csv'
        with open(name, 'w') as fid:
            writer = csv.writer(fid)
            for i in range(self.ui.number_span.value()):
                writer.writerow([self.ui.BridgeProperties.item(i,0).text(), self.ui.BridgeProperties.item(i,1).text(), self.ui.BridgeProperties.item(i,2).text(),self.ui.BridgeProperties.item(i,3).text(), self.ui.BridgeProperties.item(i,4).text()])
    # signal for import csv file
    def import_bridge_csv(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self,'Import bridge data from csv file',filter=' All files (*);;All csv files (*.csv)', initialFilter='All csv files (*.csv)')
        aux = []
        if filename[0]:
            with open(filename[0],'r') as fid:
                reader = csv.reader(fid)
                for row in reader:
                    aux.append(row)
            if  int(reader.line_num) == 1:
                self.ui.bridge_types.setCurrentIndex(1)
            elif int(reader.line_num) > 1:
                self.ui.bridge_types.setCurrentIndex(2)
            elif int(reader.line_num) == 0:
                self.open_message('The csv file has not any data. Please check the csv file')
                return 0
            self.ui.number_span.setMinimum(int(reader.line_num))
            self.ui.number_span.setMaximum(int(reader.line_num))
            self.ui.number_span.setValue(int(reader.line_num))
            self.ui.BridgeProperties.setRowCount(int(reader.line_num))
            for i in range(int(reader.line_num)):    
                try:
                    if float(aux[i][0]) >= 0 and  float(aux[i][1]) >= 0 and float(aux[i][2]) >= 0 and float(aux[i][3]) >= 0 and float(aux[i][4]) >= 0:
                        self.ui.BridgeProperties.setItem(i,0,QtWidgets.QTableWidgetItem(aux[i][0]))
                        self.ui.BridgeProperties.setItem(i,1,QtWidgets.QTableWidgetItem(aux[i][1]))
                        self.ui.BridgeProperties.setItem(i,2,QtWidgets.QTableWidgetItem(aux[i][2]))
                        self.ui.BridgeProperties.setItem(i,3,QtWidgets.QTableWidgetItem(aux[i][3]))
                        self.ui.BridgeProperties.setItem(i,4,QtWidgets.QTableWidgetItem(aux[i][4]))
                    else:
                        self.open_message('Some data is lower than 0 or is not valid value. Please check the csv file and solve it. \n\nRemember ! Only real non-zero positive values are allowed. Chracters are not allowed.')
                        return 0
                except:
                    self.open_message('Some data is a character, not value. Please check the csv file and solve it. \n\nRemember ! Only real non-zero positive values are allowed. Chracters are not allowed.')
                    return 0
    #--------------------------- train data definition ----------------------
    # signal function for selectButton
    def get_selected_item_list(self,index):
        row = index.row()
        model = self.ui.list_trains.model()
        self.selectedItem = model.item(row)
        self.selectedRow = row
        return self
    def selectButton_click(self):
        try:
            if self.ui.list_selected.model().rowCount() != 0:
                list_selected = []
                for i in range(self.ui.list_selected.model().rowCount()):
                    item = self.ui.list_selected.model().item(i)
                    list_selected.append(str(item.text()))
                if str(self.selectedItem.text()) not in list_selected:
                    self.list_selected_trains.appendRow(QtGui.QStandardItem(str(self.selectedItem.text())))
            else:
                self.list_selected_trains.appendRow(QtGui.QStandardItem(str(self.selectedItem.text())))
        except:
            self.open_message('Train is not selected. Please select a train!')
    def selectAll(self):
        self.ui.list_selected.model().clear()
        for i in range(self.ui.list_trains.model().rowCount()):
            self.list_selected_trains.appendRow(QtGui.QStandardItem(str(self.ui.list_trains.model().item(i).text())))
    # signal function for remove buttons
    def get_selected_item_to_remove(self,index):
        self.remove_row = index.row()
    def removeButton_click(self):
        try:
            self.ui.list_selected.model().removeRow(self.remove_row)
        except:
            self.open_message('Please select a train to remove')
            return 0
    def removeAll(self):
        self.ui.list_selected.model().clear()
    # signal function for delect train button
    def delete_train(self):
        self.ui.list_trains.model().removeRow(self.selectedRow)
    # signal function for show train data button
    def show_train(self):
        try:
            trainname = str(self.selectedItem.text())
            tra = 0
            if  trainname in self.TrainInteList:
                Train = train(); Train.intetrain(trainname,tra,self.add_trainInte)
                self.ui.TrainTable.setColumnCount(6)
                self.ui.TrainTable.setRowCount(len(Train.dist))
                self.ui.number_axles.setMaximum(len(Train.dist))
                self.ui.number_axles.setMinimum(len(Train.dist))
                self.ui.number_axles.setValue(len(Train.dist))
                self.ui.TrainTable.setHorizontalHeaderLabels(str("Axle Dist. (m); Axle Load (N); mw (kg); mb (kg); k1 (N/m); c1 (N.s/m);").split(";"))
                for i in range(len(Train.dist)):
                    self.ui.TrainTable.setItem(i,0,QtWidgets.QTableWidgetItem(str('%.3f' % Train.dist[i])))
                    self.ui.TrainTable.setItem(i,1,QtWidgets.QTableWidgetItem(str('%.2f' % Train.peso[i])))
                    self.ui.TrainTable.setItem(i,2,QtWidgets.QTableWidgetItem(str('%.2f' % Train.mw[i])))
                    self.ui.TrainTable.setItem(i,3,QtWidgets.QTableWidgetItem(str('%.2f' % Train.mb[i])))
                    self.ui.TrainTable.setItem(i,4,QtWidgets.QTableWidgetItem(str('%.2f' % Train.k1[i])))
                    self.ui.TrainTable.setItem(i,5,QtWidgets.QTableWidgetItem(str('%.2f' % Train.c1[i])))

            else:
                Train = train(); Train.cmovtrain(trainname,tra,self.add_train)
                self.ui.TrainTable.setRowCount(len(Train.dist))
                self.ui.number_axles.setMaximum(len(Train.dist))
                self.ui.number_axles.setMinimum(len(Train.dist))
                self.ui.number_axles.setValue(len(Train.dist))
                for i in range(len(Train.dist)):
                    self.ui.TrainTable.setItem(i,0,QtWidgets.QTableWidgetItem(str('%.3f' % Train.dist[i])))
                    self.ui.TrainTable.setItem(i,1,QtWidgets.QTableWidgetItem(str('%.2f' % Train.peso[i])))
        except:
            self.open_message("Train is not selected. Please select a train in Available trains")
            return 0
        return
    # signal function for save train button
    def save_train(self):
        if self.ui.number_axles.value() == 0:
            self.open_message("There is not values in the Train Data's table. Please provide data!")
            return 0
        else:
            self.new_train_table.show()
    def  saveNewTrain(self):
        if  str(self.new_train_table.lineedit1.text()) != "Enter a name":
            self.newTrainName = str(self.new_train_table.lineedit1.text())
        else:
            self.open_message("Please provide a name for the train!")
            return 0
        if str(self.new_train_table.lineedit2.text()) != "Enter a file name":
            self.newTrainFileName = str(self.new_train_table.lineedit2.text())
        else:
            self.open_message("Please provide a file name for saving")
            return 0
        if str(self.new_train_table.lineedit3.text()) != "Enter a value":
            self.newTrainVmax = float(str(self.new_train_table.lineedit3.text()))
        else:
            self.open_message("Please provide a integer value for saving")
            return 0
        c1 = [] # Axle distance
        c2 = [] # Axle loaid
        try:
            for r in range(self.ui.number_axles.value()):
                if str(self.ui.TrainTable.item(r,0).text()) != "":
                    c1.append(float(str(self.ui.TrainTable.item(r,0).text())))
                    c2.append(float(str(self.ui.TrainTable.item(r,1).text())))
            path = getcwd()
            folder = path + '/trains'
            if  isdir(folder) is False:
                system('mkdir '+folder)
            namefile = path +'/trains/'+self.newTrainFileName + '.dat'
            np.savetxt(namefile, np.c_[c1,c2],fmt='%8.3f')
            self.ui.list_trains.model().appendRow(QtGui.QStandardItem(self.newTrainName))
            self.add_train.append((self.newTrainName,namefile,self.newTrainVmax))
            self.new_train_table.close()
        except:
            self.open_message('Train data was not defined in the table. Please provide data !')
            self.new_train_table.close()
            return 0
    # signal function for clear table
    def  clear_table(self):
        self.ui.TrainTable.clear()
        self.ui.TrainTable.setHorizontalHeaderLabels(["Axle Dist. (m)","Axle Load (N)"])
        self.ui.number_axles.setMinimum(0)
        self.ui.number_axles.setMaximum(60)
        self.ui.number_axles.setValue(0)
        return
    def adjust_train_table(self):
        self.ui.TrainTable.setRowCount(self.ui.number_axles.value())
        for i in range(self.ui.number_axles.value()):
            for j in range(2):
                self.ui.TrainTable.setItem(i,j, QtWidgets.QTableWidgetItem())
        return
    #signal for load train file
    def load_train_file(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self,'Load train\'s file',filter=' All files (*);;All text files (*.dat);;All csv files (*.csv)', initialFilter='All files (*)')
        if filename:
            self.AddTrainFileName = str(filename[0])
            if  str(filename[0]).endswith('.dat'):
                a = np.loadtxt(str(filename[0]))
                self.ui.number_axles.setMinimum(0)
                self.ui.number_axles.setMaximum(60)
                self.ui.number_axles.setValue(len(a))
                for i in range(len(a)):
                    self.ui.TrainTable.setItem(i,0,QtWidgets.QTableWidgetItem(str('%.3f' % a[i,0])))
                    self.ui.TrainTable.setItem(i,1,QtWidgets.QTableWidgetItem(str('%.2f' % a[i,1])))
                self.add_train_dialog.show()
            elif str(filename[0]).endswith('.csv'):
                with open(filename[0],'rb') as fid:
                    reader = csv.reader(fid)
                    aux = []
                    for row in reader:
                        aux.append(row)
                self.ui.number_axles.setMinimum(0)
                self.ui.number_axles.setMaximum(60)
                self.ui.number_axles.setValue(len(aux))
                for i in range(len(aux)):
                    self.ui.TrainTable.setItem(i,0,QtWidgets.QTableWidgetItem(aux[i][0]))
                    self.ui.TrainTable.setItem(i,1,QtWidgets.QTableWidgetItem(aux[i][1]))
                self.add_train_dialog.show()
        return
    #--
    def  saveAddTrain(self):
        if  str(self.add_train_dialog.lineedit1.text()) != "Enter a name":
            self.AddTrainName = str(self.add_train_dialog.lineedit1.text())
        else:
            self.open_message("Please provide a name for the train!")
            return 0
        if str(self.add_train_dialog.lineedit3.text()) != "Enter a value":
            self.AddTrainVmax = float(str(self.add_train_dialog.lineedit3.text()))
        else:
            self.open_message("Please provide a integer value for saving")
            return 0
        c1 = [] # Axle distance
        c2 = [] # Axle load
        try:
            self.ui.list_trains.model().appendRow(QtGui.QStandardItem(self.AddTrainName))
            self.add_train.append((self.AddTrainName,self.AddTrainFileName,self.AddTrainVmax))
            self.add_train_dialog.close()
        except:
            self.open_message('File of train data was not loaded correctly. Please reselect the file !')
            self.add_train_dialog.close()
            return 0
        return
    #
    def saveTrainInter(self):
        self.newTrainInter.show()
        return
    def saveNewTrainInter(self):
        if  str(self.newTrainInter.lineedit1.text()) != "Enter a name":
            self.newTrainInterName = str(self.newTrainInter.lineedit1.text())
        else:
            self.open_message("Please provide a name for the train!")
            return 0
        if str(self.newTrainInter.lineedit2.text()) != "Enter a file name":
            self.newTrainInterFileName = str(self.newTrainInter.lineedit2.text())
        else:
            self.open_message("Please provide a file name for saving")
            return 0
        if str(self.newTrainInter.lineedit3.text()) != "Enter a value":
            try:
                self.newTrainInterVmax = float(str(self.newTrainInter.lineedit3.text()))
            except:
                self.open_message("Please provide a value, not a string !")
                return 0
        else:
            self.open_message("Please provide a integer value for saving")
            return 0
        c1 = [] # Axle distance
        c2 = [] # Axle load
        c3 = [] # mr - wheel mass
        c4 = [] # mb - bogie mass
        c5 = [] # k1 - prim. susp. stiffness
        c6 = [] # c1 - prim. susp. damping
        try:
            for r in range(self.newTrainInter.table.rowCount()):
                if str(self.newTrainInter.table.item(r,0).text()) != "":
                    c1.append(float(str(self.newTrainInter.table.item(r,0).text())))
                    c2.append(float(str(self.newTrainInter.table.item(r,1).text())))
                    c3.append(float(str(self.newTrainInter.table.item(r,2).text())))
                    c4.append(float(str(self.newTrainInter.table.item(r,3).text())))
                    c5.append(float(str(self.newTrainInter.table.item(r,4).text())))
                    c6.append(float(str(self.newTrainInter.table.item(r,5).text())))
            path = getcwd()
            folder = path + '/trains'
            if  isdir(folder) is False:
                system('mkdir '+folder)
            namefile = path +'/trains/'+self.newTrainInterFileName + '.dat'
            np.savetxt(namefile, np.c_[c1,c2,c3,c4,c5,c6],fmt='%8.3f')
            self.ui.list_trains.model().appendRow(QtGui.QStandardItem(self.newTrainInterName))
            self.add_trainInte.append((self.newTrainInterName,namefile,self.newTrainInterVmax))
            self.TrainInteList.append(self.newTrainInterName)
            self.newTrainInter.close()
        except:
            self.open_message('Train data was not defined in the table. Please provide data !')
            return 0
        self.newTrainInter.close()
        return
    #
    def opencsvfile(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self,'Load train\'s file',filter=' All files (*);;All text files (*.dat);;All csv files (*.csv)', initialFilter='All files (*)')
        if filename[0]:
            if  str(filename[0]).endswith('.dat'):
                a = np.loadtxt(str(filename[0]))
                self.newTrainInter.table.setRowCount(len(a))
                for i in range(len(a)):
                    self.newTrainInter.table.setItem(i,0,QtWidgets.QTableWidgetItem(str('%.3f' % a[i,0])))
                    self.newTrainInter.table.setItem(i,1,QtWidgets.QTableWidgetItem(str('%.2f' % a[i,1])))
                    self.newTrainInter.table.setItem(i,2,QtWidgets.QTableWidgetItem(str('%.2f' % a[i,2])))
                    self.newTrainInter.table.setItem(i,3,QtWidgets.QTableWidgetItem(str('%.2f' % a[i,3])))
                    self.newTrainInter.table.setItem(i,4,QtWidgets.QTableWidgetItem(str('%.2f' % a[i,4])))
                    self.newTrainInter.table.setItem(i,5,QtWidgets.QTableWidgetItem(str('%.2f' % a[i,5])))
            elif str(filename[0]).endswith('.csv'):
                with open(filename[0],'rb') as fid:
                    reader = csv.reader(fid)
                    aux = []
                    for row in reader:
                        aux.append(row)
                self.newTrainInter.table.setRowCount(len(aux))
                for i in range(len(aux)):
                    self.newTrainInter.table.setItem(i,0,QtWidgets.QTableWidgetItem(aux[i][0]))
                    self.newTrainInter.table.setItem(i,1,QtWidgets.QTableWidgetItem(aux[i][1]))
                    self.newTrainInter.table.setItem(i,2,QtWidgets.QTableWidgetItem(aux[i][2]))
                    self.newTrainInter.table.setItem(i,3,QtWidgets.QTableWidgetItem(aux[i][3]))
                    self.newTrainInter.table.setItem(i,4,QtWidgets.QTableWidgetItem(aux[i][4]))
                    self.newTrainInter.table.setItem(i,5,QtWidgets.QTableWidgetItem(aux[i][5]))
        return
    # signal function to clear data of train interaction dialog
    def clearTrainInteData(self):
        self.newTrainInter.table.clear()
        self.newTrainInter.table.setRowCount(70)
        self.newTrainInter.table.setColumnCount(6)
        self.newTrainInter.table.setHorizontalHeaderLabels(str("Axle Dist. (m); Axle Load (N); mw (kg); mb (kg); k1 (N/m); c1 (N.s/m);").split(";"))
        # add item for table
        for i in range(70):
            for j in range(6):
                self.newTrainInter.table.setItem(i,j, QtWidgets.QTableWidgetItem())
        self.newTrainInter.lineedit1.setText("Enter a name")
        self.newTrainInter.lineedit2.setText("Enter a file name")
        self.newTrainInter.lineedit3.setText("Enter a value")
        return
    # signal function for Submit slected trains button
    def submitTrains(self):
        # Check the list of trains used for calculation
        if self.ui.list_selected.model().rowCount()==0:
            self.open_message('No trains was selected for analysis. Please select trains!')
            self.ui.TotalTrainsSum.setText('No selected trains')
            self.ui.Tstatus.setText("UNCOMPLETED")
            return 0
        else:
            self.ui.TotalTrainsSum.setText(str(self.ui.list_selected.model().rowCount()))
            self.ui.Tstatus.setText("COMPLETED")
            self.TrainNames = []
            list_selected_trains = QtGui.QStandardItemModel()
            self.ui.listTrainsResults.setModel(list_selected_trains)
            for i in range(self.ui.list_selected.model().rowCount()):
                item=self.ui.list_selected.model().item(i)
                self.TrainNames.append(str(item.text()))
                list_selected_trains.appendRow(QtGui.QStandardItem(str(item.text())))
                self.ui.comboSelectedTrains.addItem("")
                self.ui.comboSelectedTrains.setItemText(i+1, str(item.text()))
        return
    def deleteSubmittedTrains(self):
        self.ui.TotalTrainsSum.setText(" ")
        self.ui.Tstatus.setText("DELETED")
        self.TrainNames = []
        return
    #--------------------------- Analysis options ----------------------
    # set warning  message function for interaction analysis
    def messageInter(self):
        if self.ui.interaction.isChecked():
            if self.ui.label_35.text() != "LOADED":
                self.warning_message('You have just selected the interaction analysis. This option requires the definition of 1/4 bogie train model. \n\nPlease revise if this train model has been created in "Train data" tab.\n\nFurthermore, it is not recommended to use parallel computing with this type of analysis.')
    def messageDER(self):
        if self.ui.DER.isChecked():
            if self.ui.label_35.text() != "LOADED":
                self.warning_message('This option only determines the maximum acceleration and DAF. There is not time history.')
    def messageLIR(self):
        if self.ui.LIR.isChecked():
            if self.ui.label_35.text() != "LOADED":
                self.warning_message('This option only determines the maximum acceleration and DAF. There is not time history.')
    # set number cores when parallel button is checked
    def setNumberCores(self):
        if self.ui.ParallelComp.isChecked():
            self.ui.numberCores.setMinimum(1)
            self.ui.numberCores.setMaximum(20)
        else:
            self.ui.numberCores.setMaximum(1)
        return
    # set number cores when load distribution is checked
    def setSleeperSeperation(self):
        if self.ui.loadDist.isChecked():
            if self.ui.label_35.text() != "LOADED":
                self.warning_message('Please provide the sleeper separation!')
            self.ui.sleeper_separation.setMinimum(0.10)
            self.ui.sleeper_separation.setMaximum(1.20)
        else:
            self.ui.sleeper_separation.setMinimum(0.60)
            self.ui.sleeper_separation.setMaximum(0.60)
        return
    def setDistance(self):
        if self.ui.changeMonitoredPoint.isChecked():
            if self.ui.label_35.text() != "LOADED":
                self.warning_message('Please provide the monitored point distance! \n\nIt is noted that the monitored point distance is counted from the start point of the first span to the last span.')
            self.ui.MPdistance.setMinimum(0.)
            self.ui.MPdistance.setMaximum(1000.)
        else:
            self.ui.MPdistance.setMaximum(0.)
        return
    # signal function for save options button
    def saveoptions(self):
        self.AnalysisOptions = cell_struct()
        # check the type of dynamic analysis
        if  self.ui.movingloads.isChecked() == True:
            self.type_analysis = 1  # moving loads 
        elif self.ui.interaction.isChecked() == True:
            self.type_analysis = 2  # interaction analysis
            #self.warning_message('This type of analysis requires a definition of 1/4 bogie model for train. Please check if the train model is defined previously. \n\nIn negavitve case, please provide the train model data for interaction analysis in the Tab "Train data"! ')            
        elif self.ui.DER.isChecked() == True:
            #self.warning_message('The DER Method can be used only for the simply-supported bridge. Please revise the bridge data!')
            self.type_analysis = 3  # DER method
        elif self.ui.LIR.isChecked() == True:
            #self.warning_message('The LIR Method can be used only for the simply-supported bridge. Please revise the bridge data!')
            self.type_analysis = 4
        self.AnalysisOptions.type_analysis = self.type_analysis
        # check the train speed options
        try:
            self.vmin = float(self.ui.minspeed.text())
            self.vmax = float(self.ui.maxspeed.text())
            self.incr = float(self.ui.speedincr.text())
            if  self.vmin > self.vmax:
                self.open_message('The proposed minimum velocity is greater than the maximum velocity. Please provide another value!')
                return 0
            self.ui.MinSpeedSum.setText(str(self.vmin)+' km/h')
            self.ui.MaxSpeedSum.setText(str(self.vmax)+ ' km/h')
            self.ui.SpeedIncrSum.setText(str(self.incr)+ ' km/h')
            velo = np.arange(self.vmin,self.vmax+self.incr,self.incr)
            for i in range(len(velo)):
                self.ui.comboVelo.addItem("")
                self.ui.comboVelo.setItemText(i+1, str('%.1f' % (velo[i])))
            self.AnalysisOptions.vmin = self.vmin
            self.AnalysisOptions.vmax = self.vmax
            self.AnalysisOptions.incr = self.incr
        except:
            self.open_message('At least a value of train speed options is not real number. Please check and provide another value!')
            return 0
        # check the modal analysis options
        self.BridgeData.nmod = self.ui.numberModes.value() # number of modes
        self.dt = self.ui.timeIncr.value() # time increment for dynamic analysis
        self.ui.NumModesSum.setText(str(self.BridgeData.nmod))
        self.ui.TimeStepSum.setText(str(self.dt) + ' s')
        self.AnalysisOptions.nmod = self.ui.numberModes.value()
        self.AnalysisOptions.dt = self.dt
        # check advanced options
        # load distribution
        if self.ui.loadDist.isChecked() is False:
            self.tra = 0.
        else:
            self.tra = self.ui.sleeper_separation.value()
        self.AnalysisOptions.tra = self.tra
        # change monitored point
        if  self.ui.changeMonitoredPoint.isChecked():
            if self.ui.Bstatus_summary.text() == 'COMPLETED':
                if self.ui.MPdistance.value() > np.sum(self.BridgeData.L):
                    self.open_message('The proposed value for monitored point is outside of span length of the bridge. Please provide another correct value !')
                    self.ui.Astatus.setText('UNCOMPLETED')
                    return 0
                else:
                    self.BridgeData.x = self.ui.MPdistance.value()
            else: 
                self.open_message('The bridge data is not yet defined. Please define the bridge data before selecting this option!')
                self.ui.Astatus.setText('UNCOMPLETED')
                return 0
        # check the parallel computing options
        if self.ui.ParallelComp.isChecked():
            self.nc = self.ui.numberCores.value()
            #self.warning_message('Choosing this option will deactive the progress bar.')
        else:
            self.nc = 1
        self.AnalysisOptions.nc = self.nc
        # ----
        self.ui.Astatus.setText('COMPLETED')
        return
    def clearoptions(self):
        self.ui.Astatus.setText('DELETED')
        self.ui.TimeStepSum.setText(' ')
        self.ui.NumModesSum.setText(' ')
        self.ui.MinSpeedSum.setText(' ')
        self.ui.MaxSpeedSum.setText(' ')
        self.ui.SpeedIncrSum.setText(' ')
        self.ui.minspeed.setText('Enter value')
        self.ui.maxspeed.setText('Enter value')
        self.ui.speedincr.setText('Enter value')
        self.ui.ParallelComp.setChecked(False)
        self.ui.numberCores.setValue(1)
        self.ui.MPdistance.setValue(0.00)
        self.ui.sleeper_separation.setValue(0.00)
        self.ui.changeMonitoredPoint.setChecked(False)
        self.ui.loadDist.setChecked(False)
        return
    #------- Submit Job --------
    def submitjob(self):
        # set the progressbar to initial position
        self.ui.progressBar.setValue(0)
        # check name of project
        if len(self.ui.project_name.text()) == 0:
            self.open_message('Project Name was not defined. Please provide a name!')
            return 0
        if self.ui.Bstatus_summary.text() != "COMPLETED":
            self.open_message('The bridge data was not defined correctly. Please revise the bridge data!')
            return 0
        if  self.ui.Tstatus.text() != "COMPLETED":
            self.open_message('The train data was not defined correctly. Please revise the train data!')
            return 0
        if self.ui.Astatus.text() != "COMPLETED":
            self.open_message('The analysis options were not defined correctly. Please revise the analysis data!')
            return 0
        # Create the train velocity range
        self.velo = cell_struct()
        self.velo.ini = self.vmin
        self.velo.final = self.vmax
        self.velo.inc = self.incr
        # print(self.add_train)
        # ---- Calculation step ---------
        if self.ui.movingloads.isChecked():
            self.ui.label_35.setText('Calculating now...')
            cmovisoe(self.TrainNames, self.BridgeData, self.dt, self.velo, self.tra,1,1,0,1,self.nc,0,self.project_name,self.add_train,self.ui.progressBar)
        elif self.ui.interaction.isChecked():
            for i in self.TrainNames:
                if i not in self.TrainInteList:
                    self.open_message('Some of selected trains is not correctly defined for the dynamic analysis with interaction option. Please check the selected trains\'s list at the "Train Data" Tab!')
                    self.ui.label_35.setText('ABORTED')
                    return 0
                state = True
            if  state:
                self.ui.label_35.setText('Calculating now...')
                inteiso(self.TrainNames, self.BridgeData, self.dt, self.velo, self.tra,1,1,0,1,self.nc,0,self.project_name,self.add_trainInte,self.ui.progressBar)
        elif self.ui.DER.isChecked():
            self.ui.label_35.setText('Calculating now...')
            cder(self.BridgeData,self.TrainNames, self.velo, self.tra, 1, self.project_name, self.add_train, self.ui.progressBar)
        elif self.ui.LIR.isChecked():
            self.ui.label_35.setText('Calculating now...')
            clir(self.BridgeData,self.TrainNames, self.velo, self.tra, 1, self.project_name, self.add_train, self.ui.progressBar)
        self.ui.spinModeNumber.setMinimum(1)
        self.ui.spinModeNumber.setMaximum(self.BridgeData.nmod)
        # return information status
        self.ui.label_35.setText('COMPLETED')
        self.warning_message('Dynamic calculation is finished. REMEMBER to SAVE PROJECT NOW in order to avoid losing any input and output data of the project.')
        return
    # ---------- Global results ---------
    # signal function 
    def get_selected_item_listTrainsResults(self,index):
        row = index.row()
        model = self.ui.listTrainsResults.model()
        self.selectedItemTrain = model.item(row)
        self.selectedRowTrain = row
        return self
    def IncludeTrain_click(self):
        try:
            if self.ui.plotTrains.model().rowCount() != 0:
                lista = []
                for i in range(self.ui.plotTrains.model().rowCount()):
                    item =self.ui.plotTrains.model().item(i)
                    lista.append(str(item.text()))
                if str(self.selectedItemTrain.text()) not in lista:
                    self.plot_selected_trains.appendRow(QtGui.QStandardItem(str(self.selectedItemTrain.text())))
            else:
                self.plot_selected_trains.appendRow(QtGui.QStandardItem(str(self.selectedItemTrain.text())))
        except:
            self.open_message('Train is not selected. Please select a train!')
    def removeTrain_click(self):
        try:
            self.ui.plotTrains.model().removeRow(self.remove_row)
        except:
            self.open_message('Please select a train to remove')
            return 0
    def plotResults(self):
        if self.ui.label_35.text() != "COMPLETED":
            if self.ui.label_35.text() != "LOADED":
                self.open_message('The dynamic calculation has not been performed. Please submit the job in order to plot the results!')
                return 0
        if self.ui.plotTrains.model().rowCount() == 0:
            self.open_message('No train was included for plotting. Please include trains from the list of selected train!')
            return 0
        if self.type_analysis == 1:
            if  self.tra != 0:
                name = '-mld-'
            else:
                name = '-ml-'
        elif  self.type_analysis ==2:
            if  self.tra != 0:
                name = '-imd-'
            else:
                name = '-im-'
        TrainNames = []
        for i in range(self.ui.plotTrains.model().rowCount()):
            item=self.ui.plotTrains.model().item(i)
            TrainNames.append(str(item.text()))
        # clear old graphics
        self.plotGR.axes1.cla()
        self.plotGR.axes2.cla()
        aux1, aux2, aux3, aux4 = [],[],[],[];
        for i in TrainNames:
            path = getcwd()
            if  self.type_analysis == 1:
                tren = train(); tren.cmovtrain(i,self.tra,self.add_train);
                if  self.velo.final <= tren.vmax:
                    filename = path + '/' + self.project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(self.BridgeData.alpha)+'-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (self.BridgeData.wn[0]/2/np.pi)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-nm'+str(self.BridgeData.nmod) +'-v'+str(self.velo.ini)+'-'+str(self.velo.final) + '.dat'
                else:
                    filename = path + '/' + self.project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(self.BridgeData.alpha)+'-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (self.BridgeData.wn[0]/2/np.pi)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-nm'+str(self.BridgeData.nmod) +'-v'+str(self.velo.ini)+'-'+str(tren.vmax) + '.dat'
                dato = np.loadtxt(filename)
                self.plotGR.axes1.plot(dato[:,0],dato[:,3],label=i)
                self.plotGR.axes2.plot(dato[:,0],dato[:,4],label=i)
                amax = np.max(dato[:,3]); aux1.append(amax)
                vamax = dato[np.argmax(dato[:,3]),0]; aux2.append(vamax)
                dmax = np.max(dato[:,4]); aux3.append(dmax)
                vdmax = dato[np.argmax(dato[:,4]),0]; aux4.append(vdmax)
            elif  self.type_analysis == 2:
                tren = train(); tren.intetrain(i,self.tra,self.add_trainInte);
                if  self.velo.final <= tren.vmax:
                    filename = path + '/' + self.project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(self.BridgeData.alpha)+'-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (self.BridgeData.wn[0]/2/np.pi)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-nm'+str(self.BridgeData.nmod) +'-v'+str(self.velo.ini)+'-'+str(self.velo.final) + '.dat'
                else:
                    filename = path + '/' + self.project_name + '/' + tren.nombre+'-envelope'+ name + 'skew'+str(self.BridgeData.alpha)+'-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (self.BridgeData.wn[0]/2/np.pi)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-nm'+str(self.BridgeData.nmod) +'-v'+str(self.velo.ini)+'-'+str(tren.vmax) + '.dat'
                dato = np.loadtxt(filename)
                self.plotGR.axes1.plot(dato[:,0],dato[:,3],label=i)
                self.plotGR.axes2.plot(dato[:,0],dato[:,4],label=i)
                amax = np.max(dato[:,3]); aux1.append(amax)
                vamax = dato[np.argmax(dato[:,3]),0]; aux2.append(vamax)
                dmax = np.max(dato[:,4]); aux3.append(dmax)
                vdmax = dato[np.argmax(dato[:,4]),0]; aux4.append(vdmax)
            elif self.type_analysis == 4:
                tren = train(); tren.cmovtrain(i,self.tra,self.add_train);
                f0 = np.pi*np.sqrt(self.BridgeData.EI/self.BridgeData.m)/(2.*self.BridgeData.L**2)
                if  self.velo.final <= tren.vmax:
                    filename = path + '/' + self.project_name + '/' + tren.nombre+'-LIR-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (f0)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-v'+str(self.velo.ini)+'-'+str(self.velo.final) + '.dat'
                else:
                    filename = path + '/' + self.project_name + '/' + tren.nombre+'-LIR-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (f0)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-v'+str(self.velo.ini)+'-'+str(tren.vmax) + '.dat'
                dato = np.loadtxt(filename)
                self.plotGR.axes1.plot(dato[:,0],dato[:,2],label=i)
                self.plotGR.axes2.plot(dato[:,0],dato[:,3],label=i)
                amax = np.max(dato[:,2]); aux1.append(amax)
                vamax = dato[np.argmax(dato[:,2]),0]; aux2.append(vamax)
                dmax = np.max(dato[:,3]); aux3.append(dmax)
                vdmax = dato[np.argmax(dato[:,3]),0]; aux4.append(vdmax)
            elif self.type_analysis == 3:
                tren = train(); tren.cmovtrain(i,self.tra,self.add_train);
                f0 = np.pi*np.sqrt(self.BridgeData.EI/self.BridgeData.m)/(2.*self.BridgeData.L**2)
                if  self.velo.final <= tren.vmax:
                    filename = path + '/' + self.project_name + '/' + tren.nombre+'-DER-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (f0)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-v'+str(self.velo.ini)+'-'+str(self.velo.final) + '.dat'
                else:
                    filename = path + '/' + self.project_name + '/' + tren.nombre+'-DER-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (f0)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-v'+str(self.velo.ini)+'-'+str(tren.vmax) + '.dat'
                dato = np.loadtxt(filename)
                self.plotGR.axes1.plot(dato[:,0],dato[:,2],label=i)
                self.plotGR.axes2.plot(dato[:,0],dato[:,3],label=i)
                amax = np.max(dato[:,2]); aux1.append(amax)
                vamax = dato[np.argmax(dato[:,2]),0]; aux2.append(vamax)
                dmax = np.max(dato[:,3]); aux3.append(dmax)
                vdmax = dato[np.argmax(dato[:,3]),0]; aux4.append(vdmax)
        #    
        self.plotGR.axes1.legend(loc=0,ncol=3)
        self.plotGR.axes1.set_xlabel(r'Velocity (km/h)')
        self.plotGR.axes1.set_ylabel(r'Acceleration (m/s$^2$)')
        self.plotGR.axes1.grid()
        self.plotGR.axes2.legend(loc=0,ncol=3)
        self.plotGR.axes2.set_xlabel(r'Velocity (km/h)')
        self.plotGR.axes2.set_ylabel(r'DAF ($1+\varphi$)')
        self.plotGR.axes2.grid()
        self.plotGR.draw()
        # text view for maximum accelearation and displacment 
        lf.ui.label_70.setText(str('%.2f' % (max(aux1))))
        self.ui.label_75.setText(str('%.2f' % (aux2[np.argmax(np.array(aux1))])))
        self.ui.label_73.setText(str('%.2f' % (max(aux3))))
        self.ui.label_77.setText(str('%.2f' % (aux4[np.argmax(np.array(aux3))])))
    def saveFigures(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self,filter=' All files (*);;Portable Network Graphics (*.png)', initialFilter='Portable Network Graphics (*.png)')
        if str(filename[0]).endswith('.png'):
            name=str(filename[0])
        else:
            name = str(filename[0]) + '.png'
        self.plotGR.fig.savefig(name,dpi=300)
        return
    def plotMaxAccelLimit(self):
        try:
            if  self.ui.slabTrack.isChecked():
                self.plotGR.axes1.plot([self.vmin,self.vmax],[5.,5.],'r',label='Accel. limit')
                self.plotGR.axes1.legend(loc=0,ncol=3)
                self.plotGR.draw()
            elif self.ui.ballastTrack.isChecked():
                self.plotGR.axes1.plot([self.vmin,self.vmax],[ 3.5,3.5],'k--',label='Accel. limit')
                self.plotGR.axes1.legend(loc=0,ncol=3)
                self.plotGR.draw()
        except: 
            self.open_message('Project data was not defined. Please check all data!')
            return 0
        return
    def plotUIC(self):
        if  self.BridgeData.nv > 1 :
            self.open_message('This option is only used for simply-supported bridge.')
            return 0
        else:
            try:
                velo = np.arange(self.vmin,self.vmax+self.incr,self.incr)
                K = velo/3.6/(2*self.BridgeData.L*self.BridgeData.wn[0]/2/np.pi)
                phi_dyn = 1+ K/(1-K+K**4)
                self.plotGR.axes2.plot(velo,phi_dyn,'k--',label=r'$1+\varphi_{UIC}^{\prime}$')
                self.plotGR.axes2.legend(loc=0,ncol=3)
                self.plotGR.draw()  
            except:
                self.open_message('Project data was not defined. Please check all data!')
                return 0
        return
    #-----------------------------------
    # --- History Results functions ----
    #-----------------------------------
    def plotAccelTF(self):
        if  self.ui.comboSelectedTrains.currentIndex() != 0 and self.ui.comboVelo.currentIndex() != 0:
            if self.type_analysis == 1:
                tren = train(); tren.cmovtrain(str(self.ui.comboSelectedTrains.currentText()),self.tra,self.add_train)
                if  self.tra != 0:
                    name = '-mld-'
                else:
                    name = '-ml-'
                if float(self.ui.comboVelo.currentText()) <= tren.vmax:
                    path=getcwd()
                    datafile = path + '/' + self.project_name + '/' + tren.nombre+ name + 'skew'+str(self.BridgeData.alpha)+'-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (self.BridgeData.wn[0]/2/np.pi)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-nm'+str(self.BridgeData.nmod) +'-v'+str(self.ui.comboVelo.currentText())+'.dat'
                    dato = np.loadtxt(datafile)
                    iamax = np.argmax(abs(dato[:,3])); tamax = dato[iamax,0]; amax = dato[iamax,3];
                    leyenda = r'amax = %.3f m/s$^2$ at t = %.3f s' % (amax,tamax)
                    tmedian = 0.5*(dato[0,0]+dato[-1,0])
                    self.plotHR.axes1.cla()
                    self.plotHR.axes1.plot(dato[:,0],dato[:,3])
                    self.plotHR.axes1.plot(tamax,amax,'or',ms=5.)
                    if tamax > tmedian:
                        self.plotHR.axes1.text(tamax-0.05, amax,leyenda,horizontalalignment='right')
                    else:
                        self.plotHR.axes1.text(tamax+0.05, amax,leyenda,horizontalalignment='left')
                    self.plotHR.axes1.set_xlabel(r'Time (s)')
                    self.plotHR.axes1.set_xlim([dato[0,0],dato[-1,0]])
                    self.plotHR.axes1.set_ylabel(r'Acceleration (m/s$^2$)')
                    self.plotHR.axes1.grid()
                    # plot result in frequency domain
                    freq,value = funcionFFT(np.c_[dato[:,0],dato[:,3]])
                    self.plotHR.axes2.cla()
                    self.plotHR.axes2.semilogx(freq,value)
                    self.plotHR.axes2.set_xlabel(r'Frequency (Hz)')
                    self.plotHR.axes2.set_ylabel(r'Amplitude (m/s$^2$/Hz)')
                    self.plotHR.axes2.grid()
                    self.plotHR.draw()
                else:
                    self.open_message('The selected velocity is greater than the maximum train velocity')
            elif self.type_analysis == 2:
                tren = train(); tren.intetrain(str(self.ui.comboSelectedTrains.currentText()),self.tra,self.add_trainInte)
                if  self.tra != 0:
                    name = '-imd-'
                else:
                    name = '-im-'
                if float(self.ui.comboVelo.currentText()) <= tren.vmax:
                    path=getcwd()
                    datafile = path + '/' + self.project_name + '/' + tren.nombre+ name + 'skew'+str(self.BridgeData.alpha)+'-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (self.BridgeData.wn[0]/2/np.pi)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-nm'+str(self.BridgeData.nmod) +'-v'+str(self.ui.comboVelo.currentText())+'.dat'
                    dato = np.loadtxt(datafile)
                    iamax = np.argmax(abs(dato[:,3])); tamax = dato[iamax,0]; amax = dato[iamax,3];
                    leyenda = r'amax = %.3f m/s$^2$ at t = %.3f s' % (amax,tamax)
                    tmedian = 0.5*(dato[0,0]+dato[-1,0])
                    self.plotHR.axes1.cla()
                    self.plotHR.axes1.plot(dato[:,0],dato[:,3])
                    self.plotHR.axes1.plot(tamax,amax,'or',ms=5.)
                    if tamax > tmedian:
                        self.plotHR.axes1.text(tamax-0.05, amax,leyenda,horizontalalignment='right')
                    else:
                        self.plotHR.axes1.text(tamax+0.05, amax,leyenda,horizontalalignment='left')
                    self.plotHR.axes1.set_xlabel(r'Time (s)')
                    self.plotHR.axes1.set_xlim([dato[0,0],dato[-1,0]])
                    self.plotHR.axes1.set_ylabel(r'Acceleration (m/s$^2$)')
                    self.plotHR.axes1.grid()
                    # plot result in frequency domain
                    freq,value = funcionFFT(np.c_[dato[:,0],dato[:,3]])
                    self.plotHR.axes2.cla()
                    self.plotHR.axes2.semilogx(freq,value)
                    self.plotHR.axes2.set_xlabel(r'Frequency (Hz)')
                    self.plotHR.axes2.set_ylabel(r'Amplitude (m/s$^2$/Hz)')
                    self.plotHR.axes2.grid()
                    self.plotHR.draw()
                else:
                    self.open_message('The selected velocity is greater than the maximum train velocity')
            elif self.type_analysis == 4:
                self.warning_message('The RIL method has been selected for dynamic calculation. Therefore, there are not history results to plot.')
            elif self.type_analysis ==3:
                self.warning_message('The DER method has been selected for dynamic calculation. Therefore, there are not history results to plot.')
        else:
            self.open_message('Train and its velocity were not selected. Please select the train and  its velocity!')

        return
    #----------
    def plotDispTF(self):
        if  self.ui.comboSelectedTrains.currentIndex() != 0 and self.ui.comboVelo.currentIndex() != 0:
            if self.type_analysis == 1:
                tren = train(); tren.cmovtrain(str(self.ui.comboSelectedTrains.currentText()),self.tra,self.add_train)
                if  self.tra != 0:
                    name = '-mld-'
                else:
                    name = '-ml-'
                if float(self.ui.comboVelo.currentText()) <= tren.vmax:
                    path=getcwd()
                    datafile = path + '/' + self.project_name + '/' + tren.nombre+ name + 'skew'+str(self.BridgeData.alpha)+'-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (self.BridgeData.wn[0]/2/np.pi)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-nm'+str(self.BridgeData.nmod) +'-v'+str(self.ui.comboVelo.currentText())+'.dat'
                    dato = np.loadtxt(datafile)
                    idmax = np.argmax(abs(dato[:,1])); tdmax = dato[idmax,0]; dmax = dato[idmax,1];
                    leyenda = r'dmax = %.3f mm at t = %.3f s' % (dmax*1.0e3,tdmax)
                    tmedian = 0.5*(dato[0,0]+dato[-1,0])
                    self.plotHR.axes1.cla()
                    self.plotHR.axes1.plot(dato[:,0],dato[:,1]*1.e03)
                    self.plotHR.axes1.plot(tdmax,dmax*1.e03,'or',ms=5.)
                    if tdmax > tmedian:
                        self.plotHR.axes1.text(tdmax-0.05, dmax*1e03,leyenda,horizontalalignment='right')
                    else:
                        self.plotHR.axes1.text(tdmax+0.05, dmax*1e03,leyenda,horizontalalignment='left')
                    self.plotHR.axes1.set_xlabel(r'Time (s)')
                    self.plotHR.axes1.set_xlim([dato[0,0],dato[-1,0]])
                    self.plotHR.axes1.set_ylabel(r'Displacement (mm)')
                    self.plotHR.axes1.grid()
                    # plot result in frequency domain
                    freq,value = funcionFFT(np.c_[dato[:,0],dato[:,1]*1.e3])
                    self.plotHR.axes2.cla()
                    self.plotHR.axes2.semilogx(freq,value)
                    self.plotHR.axes2.set_xlabel(r'Frequency (Hz)')
                    self.plotHR.axes2.set_ylabel(r'Amplitude (mm/Hz)')
                    self.plotHR.axes2.grid()
                    self.plotHR.draw()
                else:
                    self.open_message('The selected velocity is greater than the maximum train velocity')
                    return 0
            elif self.type_analysis == 2:
                tren = train(); tren.intetrain(str(self.ui.comboSelectedTrains.currentText()),self.tra,self.add_trainInte)
                if  self.tra != 0:
                    name = '-imd-'
                else:
                    name = '-im-'
                if float(self.ui.comboVelo.currentText()) <= tren.vmax:
                    path=getcwd()
                    datafile = path + '/' + self.project_name + '/' + tren.nombre+ name + 'skew'+str(self.BridgeData.alpha)+'-L'+str(self.BridgeData.L) + '-z'+str(self.BridgeData.xi*100) + '-f'+ str('%.2f' % (self.BridgeData.wn[0]/2/np.pi)) + 'Hz-m'+str(self.BridgeData.m/1.e03)+'t-nm'+str(self.BridgeData.nmod) +'-v'+str(self.ui.comboVelo.currentText())+'.dat'
                    dato = np.loadtxt(datafile)
                    idmax = np.argmax(abs(dato[:,1])); tdmax = dato[idmax,0]; dmax = dato[idmax,1];
                    leyenda = r'dmax = %.3f mm at t = %.3f s' % (dmax*1.0e3,tdmax)
                    tmedian = 0.5*(dato[0,0]+dato[-1,0])
                    self.plotHR.axes1.cla()
                    self.plotHR.axes1.plot(dato[:,0],dato[:,1]*1.e03)
                    self.plotHR.axes1.plot(tdmax,dmax*1.e03,'or',ms=5.)
                    if tdmax > tmedian:
                        self.plotHR.axes1.text(tdmax-0.05, dmax*1e03,leyenda,horizontalalignment='right')
                    else:
                        self.plotHR.axes1.text(tdmax+0.05, dmax*1e03,leyenda,horizontalalignment='left')
                    self.plotHR.axes1.set_xlabel(r'Time (s)')
                    self.plotHR.axes1.set_xlim([dato[0,0],dato[-1,0]])
                    self.plotHR.axes1.set_ylabel(r'Displacement (mm/s$^2$)')
                    self.plotHR.axes1.grid()
                    # plot result in frequency domain
                    freq,value = funcionFFT(np.c_[dato[:,0],dato[:,1]*1.e3])
                    self.plotHR.axes2.cla()
                    self.plotHR.axes2.semilogx(freq,value)
                    self.plotHR.axes2.set_xlabel(r'Frequency (Hz)')
                    self.plotHR.axes2.set_ylabel(r'Amplitude (mm/Hz)')
                    self.plotHR.axes2.grid()
                    self.plotHR.draw()
                else:
                    self.open_message('The selected velocity is greater than the maximum train velocity')
                    return 0
            elif self.type_analysis ==3:
                self.warning_message('The DER method has been selected for dynamic calculation. Therefore, there are not history results to plot.')
            elif self.type_analysis ==4:
                self.warning_message('The RIL method has been selected for dynamic calculation. Therefore, there are not history results to plot.')
        else:
            self.open_message('Train and its velocity were not selected. Please select the train and  its velocity!')
        #             
        return
    #---------
    def clearFigure(self):
        self.plotHR.axes1.cla()
        self.plotHR.axes1.set_xlabel(r'Time (s)')
        self.plotHR.axes2.cla()
        self.plotHR.axes2.set_xlabel(r'Frequency (Hz)')
        self.plotHR.axes2.set_ylabel(r'Amplitude')
        self.plotHR.draw()
    #-----------------------------------
    # --- Modal Modes functions ----
    #-----------------------------------
    # signal function for plot vertical mode shape
    def plotVModeShape(self):
        mode = int(self.ui.spinModeNumber.value())
        if mode == 0:
            self.open_message('No mode has been selected. Please select a nonzero value!')
        else:
            try:
                self.plotMM.axes.cla()
                if self.BridgeData.nv == 1:
                    dx = 0.1; f0 = self.BridgeData.wn[mode-1]/2/np.pi
                    param = self.BridgeData.param
                    beta = self.BridgeData.beta 
                    x = np.arange(0,self.BridgeData.L+dx,dx)
                    y = param[mode-1,0]*np.sin(beta[mode-1]*x) + param[mode-1,1]*np.cos(beta[mode-1]*x) + param[mode-1,2]*np.sinh(beta[mode-1]*x) + param[mode-1,3]*np.cosh(beta[mode-1]*x)
                    self.plotMM.axes.plot(x,y,lw=2)
                    self.plotMM.axes.plot(x[0],0,'r^',ms=10)
                    self.plotMM.axes.plot(x[-1],0,'r^',ms=10)
                    self.plotMM.axes.plot([x[0],x[-1]],[0,0],'k',lw=2)
                    self.plotMM.axes.spines['right'].set_visible(False)
                    self.plotMM.axes.spines['left'].set_visible(False)
                    self.plotMM.axes.spines['top'].set_visible(False)
                    self.plotMM.axes.spines['bottom'].set_visible(False)
                    self.plotMM.axes.set_xticks([])
                    self.plotMM.axes.set_yticks([])
                    self.plotMM.axes.set_xlim([x[0],x[-1]])
                    self.plotMM.axes.set_ylim([-1.5,1.5])
                    self.plotMM.draw()
                    self.ui.FreqValue.setText(str('%.3f' % (f0)))
                else:
                    if  self.BridgeData.portico == 1:
                        dx = 0.1; f0 = self.BridgeData.wn[mode-1]/2./np.pi
                        param = self.BridgeData.param
                        beta = self.BridgeData.beta
                        x = np.arange(0,self.BridgeData.L[1]+dx,dx); 
                        y = param[mode-1,6]*np.sin(beta[mode-1,1]*x) + param[mode-1,7]*np.cos(beta[mode-1,1]*x) + param[mode-1,8]*np.sinh(beta[mode-1,1]*x) + param[mode-1,9]*np.cosh(beta[mode-1,1]*x)
                        d1 = np.arange(0,self.BridgeData.L[0]+dx,dx)
                        inc1 = param[mode-1,0]*np.sin(beta[mode-1,0]*d1) + param[mode-1,1]*np.cos(beta[mode-1,0]*d1) + param[mode-1,2]*np.sinh(beta[mode-1,0]*d1) + param[mode-1,3]*np.cosh(beta[mode-1,0]*d1)
                        d2 = np.linspace(self.BridgeData.L[2],0,int(self.BridgeData.L[2]/dx)+1)
                        inc2 = param[mode-1,12]*np.sin(beta[mode-1,2]*d2) + param[mode-1,13]*np.cos(beta[mode-1,2]*d2) + param[mode-1,14]*np.sinh(beta[mode-1,2]*d2) + param[mode-1,15]*np.cosh(beta[mode-1,2]*d2)
                        self.plotMM.axes.plot(x,y,'b',lw=2)
                        self.plotMM.axes.plot(-inc1,-self.BridgeData.L[0] + d1, 'b', lw=2)
                        self.plotMM.axes.plot(self.BridgeData.L[1]+inc2, -d2, 'b', lw=2)
                        self.plotMM.axes.plot([0,self.BridgeData.L[1]],[0,0],'k',lw=2)
                        self.plotMM.axes.plot([0,0],[-self.BridgeData.L[0],0],'k',lw=2)
                        self.plotMM.axes.plot([self.BridgeData.L[1],self.BridgeData.L[1]],[-self.BridgeData.L[-1],0],'k',lw=2)
                        self.plotMM.axes.spines['right'].set_visible(False)
                        self.plotMM.axes.spines['left'].set_visible(False)
                        self.plotMM.axes.spines['top'].set_visible(False)
                        self.plotMM.axes.spines['bottom'].set_visible(False)
                        self.plotMM.axes.set_xticks([])
                        self.plotMM.axes.set_yticks([])
                        if self.BridgeData.boun == 0:
                            self.plotMM.axes.plot(0,-self.BridgeData.L[0],'r^',ms=10)
                            self.plotMM.axes.plot(self.BridgeData.L[1],-self.BridgeData.L[2],'r^',ms=10)
                        elif self.BridgeData.boun ==1:
                            self.plotMM.axes.plot([-0.1,0.1],[-self.BridgeData.L[0],-self.BridgeData.L[0]],'r',lw=5)
                            self.plotMM.axes.plot([self.BridgeData.L[1]-0.1,self.BridgeData.L[1]+0.1],[-self.BridgeData.L[2],-self.BridgeData.L[2]],'r',lw=5)
                        self.plotMM.draw()
                        self.ui.FreqValue.setText(str('%.3f' % (f0)))
                    elif self.BridgeData.portico == 0:
                        dx = 0.1; f0 = self.BridgeData.wn[mode-1]/2./np.pi
                        param = self.BridgeData.param
                        beta = self.BridgeData.beta
                        spans = {'span'+str(i): np.arange(0,self.BridgeData.L[i]+dx,dx) for i in range(self.BridgeData.nv)}
                        defor = {'defo'+str(i): param[mode-1,i*6]*np.sin(beta[mode-1,i]*spans['span'+str(i)])+param[mode-1,i*6+1]*np.cos(beta[mode-1,i]*spans['span'+str(i)])+param[mode-1,i*6+2]*np.sinh(beta[mode-1,i]*spans['span'+str(i)])+param[mode-1,i*6+3]*np.cosh(beta[mode-1,i]*spans['span'+str(i)]) for i in range(self.BridgeData.nv)}
                        self.plotMM.axes.plot(spans['span0'],defor['defo0'],'b',lw=2)
                        for i in range(1,self.BridgeData.nv):
                            self.plotMM.axes.plot(np.sum(self.BridgeData.L[:i])+spans['span'+str(i)], defor['defo'+str(i)],'b',lw=2)           
                        self.plotMM.axes.plot([0,np.sum(self.BridgeData.L)],[0,0],'k',lw=2)
                        for i in range(1,self.BridgeData.nv):
                            self.plotMM.axes.plot(np.sum(self.BridgeData.L[:i]),0,'r^',ms=10)
                        if self.BridgeData.boun == 0:
                            self.plotMM.axes.plot(0,0,'r^',ms=10)
                            self.plotMM.axes.plot(np.sum(self.BridgeData.L),0,'r^',ms=10)
                        elif self.BridgeData.boun ==1:
                            self.plotMM.axes.plot([0,0],[-0.1,0.1],'r',lw=5)
                            self.plotMM.axes.plot([np.sum(self.BridgeData.L),np.sum(self.BridgeData.L)],[-0.1,0.1],'r',lw=5)
                        self.plotMM.axes.spines['right'].set_visible(False)
                        self.plotMM.axes.spines['left'].set_visible(False)
                        self.plotMM.axes.spines['top'].set_visible(False)
                        self.plotMM.axes.spines['bottom'].set_visible(False)
                        self.plotMM.axes.set_xticks([])
                        self.plotMM.axes.set_yticks([])
                        self.plotMM.draw()
                        self.ui.FreqValue.setText(str('%.3f' % (f0)))
            except:
                self.open_message('The calculation has not been performed. Please complete the calculation procedure!')
                return 0
        return
    # signal function for plot torsional mode shape
    def plotTModeShape(self):
        mode = int(self.ui.spinModeNumber.value())
        if mode == 0:
            self.open_message('No mode has been selected. Please select a nonzero value!')
        else:
            try:
                self.plotMM.axes.cla()
                if self.BridgeData.nv == 1:
                    dx = 0.1; f0 = self.BridgeData.wn[mode-1]/2/np.pi
                    param = self.BridgeData.param
                    la = self.BridgeData.la
                    x = np.arange(0,self.BridgeData.L+dx,dx)
                    y = param[mode-1,4]*np.sinh(la[mode-1]*x) + param[mode-1,5]*np.cosh(la[mode-1]*x)
                    self.plotMM.axes.plot(x,y,lw=2)
                    if  self.BridgeData.boun == 0:
                        self.plotMM.axes.plot(x[0],0,'r^')
                        self.plotMM.axes.plot(x[-1],0,'r^')
                    elif self.BridgeData.boun == 1:
                        self.plotMM.axes.plot([x[0],x[0]],[-0.25,0.25])
                        self.plotMM.axes.plot([x[-1],x[-1]],[-0.25,0.25])
                    self.plotMM.axes.plot([x[0],x[-1]],[0,0],'k',lw=2)
                    self.plotMM.axes.spines['right'].set_visible(False)
                    self.plotMM.axes.spines['left'].set_visible(False)
                    self.plotMM.axes.spines['top'].set_visible(False)
                    self.plotMM.axes.spines['bottom'].set_visible(False)
                    self.plotMM.axes.set_xticks([])
                    self.plotMM.axes.set_yticks([])
                    self.plotMM.axes.set_xlim([x[0],x[-1]])
                    self.plotMM.axes.set_ylim([-1.0,1.0])
                    self.plotMM.draw()
                    self.ui.FreqValue.setText(str('%.3f' % (f0)))
                else:
                    if  self.BridgeData.portico == 1:
                        dx = 0.1; f0 = self.BridgeData.wn[mode-1]/2./np.pi
                        param = self.BridgeData.param
                        beta = self.BridgeData.beta
                        la = self.BridgeData.la
                        x = np.arange(0,self.BridgeData.L[1]+dx,dx)
                        y = param[mode-1,10]*np.sinh(la[mode-1,1]*x) + param[mode-1,11]*np.cosh(la[mode-1,1]*x)
                        self.plotMM.axes.plot(x,y,'b',lw=2)
                        self.plotMM.axes.plot([0,self.BridgeData.L[1]],[0,0],'k',lw=2)
                        self.plotMM.axes.plot([0,0],[-self.BridgeData.L[0],0],'k',lw=2)
                        self.plotMM.axes.plot([self.BridgeData.L[1],self.BridgeData.L[1]],[-self.BridgeData.L[-1],0],'k',lw=2)
                        self.plotMM.axes.spines['right'].set_visible(False)
                        self.plotMM.axes.spines['left'].set_visible(False)
                        self.plotMM.axes.spines['top'].set_visible(False)
                        self.plotMM.axes.spines['bottom'].set_visible(False)
                        self.plotMM.axes.set_xticks([])
                        self.plotMM.axes.set_yticks([])
                        if self.BridgeData.boun == 0:
                            self.plotMM.axes.plot(0,-self.BridgeData.L[0],'r^',ms=10)
                            self.plotMM.axes.plot(self.BridgeData.L[1],-self.BridgeData.L[2],'r^',ms=10)
                        elif self.BridgeData.boun ==1:
                            self.plotMM.axes.plot([-0.1,0.1],[-self.BridgeData.L[0],-self.BridgeData.L[0]],'r',lw=5)
                            self.plotMM.axes.plot([self.BridgeData.L[1]-0.1,self.BridgeData.L[1]+0.1],[-self.BridgeData.L[2],-self.BridgeData.L[2]],'r',lw=5)
                        self.plotMM.draw()
                        self.ui.FreqValue.setText(str('%.3f' % (f0)))
                    elif self.BridgeData.portico == 0:
                        dx = 0.1; f0 = self.BridgeData.wn[mode-1]/2./np.pi
                        param = self.BridgeData.param
                        la = self.BridgeData.la
                        spans = {'span'+str(i): np.arange(0,self.BridgeData.L[i]+dx,dx) for i in range(self.BridgeData.nv)}
                        defor = {'defo'+str(i): param[mode-1,i*6+4]*np.sinh(la[mode-1,i]*spans['span'+str(i)])+param[mode-1,i*6+5]*np.cosh(la[mode-1,i]*spans['span'+str(i)]) for i in range(self.BridgeData.nv)}
                        self.plotMM.axes.plot(spans['span0'],defor['defo0'],'b',lw=2)
                        for i in range(1,self.BridgeData.nv):
                            self.plotMM.axes.plot(np.sum(self.BridgeData.L[:i])+spans['span'+str(i)], defor['defo'+str(i)],'b',lw=2)           
                        self.plotMM.axes.plot([0,np.sum(self.BridgeData.L)],[0,0],'k',lw=2)                   
                        for i in range(1,self.BridgeData.nv):
                            self.plotMM.axes.plot(np.sum(self.BridgeData.L[:i]),0,'r^',ms=10)
                        for i in range(1,self.BridgeData.nv):
                            self.plotMM.axes.plot(np.sum(self.BridgeData.L[:i]),0,'r^',ms=10)
                        if self.BridgeData.boun == 0:
                            self.plotMM.axes.plot(0,0,'r^',ms=10)
                            self.plotMM.axes.plot(np.sum(self.BridgeData.L),0,'r^',ms=10)
                        elif self.BridgeData.boun ==1:
                            self.plotMM.axes.plot([0,0],[-0.1,0.1],'r',lw=5)
                            self.plotMM.axes.plot([np.sum(self.BridgeData.L),np.sum(self.BridgeData.L)],[-0.1,0.1],'r',lw=5)
                        self.plotMM.axes.spines['right'].set_visible(False)
                        self.plotMM.axes.spines['left'].set_visible(False)
                        self.plotMM.axes.spines['top'].set_visible(False)
                        self.plotMM.axes.spines['bottom'].set_visible(False)
                        self.plotMM.axes.set_xticks([])
                        self.plotMM.axes.set_yticks([])
                        self.plotMM.draw()
                        self.ui.FreqValue.setText(str('%.3f' % (f0)))
            except:
                self.open_message('The calculation has not been performed. Please complete the calculation procedure!')
                return 0
        return
    def saveModeFigures(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self,filter=' All files (*);;Portable Network Graphics (*.png)', initialFilter='Portable Network Graphics (*.png)')
        if str(filename[0]).endswith('.png'):
            name=str(filename[0])
        else:
            name = str(filename[0]) + '.png'
        self.plotMM.fig.savefig(name,dpi=300)
        return
    def clearModeShape(self):
        self.plotMM.axes.cla()
        self.plotMM.draw()
        return
    #-----------------------------------
    def closeApp(self):
        warning = QtWidgets.QMessageBox.warning(self,'CLOSING PROGRAM',"This action will close and stop the program.\nAll unsaved data will be lost.\n\nDo you really want to exit CALDINTAV 3.2?", QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)
        if warning == QtWidgets.QMessageBox.Yes:
            self.close()
    #---------------------------
    def open_message(self,texto):
        message = QtWidgets.QMessageBox.question(self,'Error',texto, QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
    def warning_message(self,texto):
        message = QtWidgets.QMessageBox.warning(self,'WARNING',texto, QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
    #--------------------------
    def open_manual(self):
        try: 
            if sys.platform == "darwin":
                filename = direction + '/user_guide.pdf'
                # PARA MAC OS SYSTEM
                system('open '+filename)
            elif  sys.platform == "linux" or "linux2":
                filename = direction + '/user_guide.pdf'
                # PARA UBUNTU-GNOME abertura PDF
                system('/usr/bin/gnome-open '+ filename)
            elif sys.platform == "win32":
                filename = direction + "\\user_guide.pdf"
                # PARA WINDOWS abertura PDF
                os.startfile(filename)
        except:
            selection_empty = QtWidgets.QMessageBox.warning(self.w, "CALDINTAV 3.22: User guide opening error","It was not possible to open the user guide due to some of this reasons:\n\n1. The rute:\n\n"+ filename + "\n\nwhere the User guide was supposed to be is missing. Please, restore the file to its location.\n\n2. You have not a PDF reader installed. Please, install one.\n\n3. Your system configuration is not compatible with the predifined commands. Please, open it manually at the folder you are given avobe.",QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            return 0
        return
    def open_about(self):
        self.about.show()
        return
    def open_license(self):
        license = LicenseDialog(self)
        license.show()
        return
    # clear all data for new project
    def clear_all(self):
        # bridge data
        self.ui.bridge_types.setCurrentIndex(0)
        self.ui.BC.setCurrentIndex(0)
        self.ui.damping_value.setText('Enter value')
        self.ui.skew_value.setText('Enter value')
        self.ui.number_span.setMinimum(0)
        self.ui.number_span.setValue(0)
        self.ui.span_summary.setText('')
        self.ui.BC_summary.setText('')
        self.ui.skew_summary.setText('')
        self.ui.Bstatus_summary.setText('')
        #  train data
        self.ui.list_trains.model().clear()
        [self.ui.list_trains.model().appendRow(QtGui.QStandardItem('HSLM-A'+str(i))) for i in range(1,11)]
        self.ui.list_selected.model().clear()
        self.ui.plotTrains.model().clear()
        self.add_train = []
        self.add_trainInte = []
        self.TrainInteList = []
        self.TrainNames = []
        self.ui.TotalTrainsSum.setText(" ")
        self.ui.Tstatus.setText("")
        self.ui.listTrainsResults.model().clear()
        self.ui.comboSelectedTrains.clear()
        self.ui.comboSelectedTrains.addItem("")
        self.ui.comboSelectedTrains.setItemText(0,"Select an item")
        self.ui.comboSelectedTrains.setCurrentIndex(0)
        # analysis data
        self.ui.Astatus.setText("")
        self.ui.TimeStepSum.setText(' ')
        self.ui.NumModesSum.setText(' ')
        self.ui.MinSpeedSum.setText(' ')
        self.ui.MaxSpeedSum.setText(' ')
        self.ui.SpeedIncrSum.setText(' ')
        self.ui.minspeed.setText('Enter value')
        self.ui.maxspeed.setText('Enter value')
        self.ui.speedincr.setText('Enter value')
        self.ui.ParallelComp.setChecked(False)
        self.ui.numberCores.setValue(1)
        self.ui.MPdistance.setValue(0.00)
        self.ui.sleeper_separation.setValue(0.00)
        self.ui.changeMonitoredPoint.setChecked(False)
        self.ui.loadDist.setChecked(False)
        self.ui.comboVelo.clear()
        self.ui.comboVelo.addItem("")
        self.ui.comboVelo.setItemText(0,"Select an item")
        self.ui.comboVelo.setCurrentIndex(0)
        # clear global plot
        self.plotGR.axes1.cla()
        self.plotGR.axes2.cla()
        self.plotGR.draw()
        self.ui.label_70.setText("")
        self.ui.label_75.setText("")
        self.ui.label_73.setText("")
        self.ui.label_77.setText("")
        # clear history plot
        self.plotHR.axes1.cla()
        self.plotHR.axes2.cla()
        self.plotHR.draw()
        # clear modes plot
        self.plotMM.axes.cla()
        self.plotMM.draw()
        self.ui.FreqValue.setText('')
        # set global status
        self.ui.lable_35.setText('')
        return
#---------------------------------------------------------------
class NewTrainDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self,parent)
        self.setWindowTitle('Save train')
        label1 = QtWidgets.QLabel("Name of the train")
        self.lineedit1 = QtWidgets.QLineEdit()
        self.lineedit1.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit1.setText("Enter a name")
        label2 = QtWidgets.QLabel("File name to save")
        self.lineedit2 = QtWidgets.QLineEdit()
        self.lineedit2.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit2.setText("Enter a file name")
        label3 = QtWidgets.QLabel("Maximum velocity of train")
        self.lineedit3 = QtWidgets.QLineEdit()
        self.lineedit3.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit3.setText("Enter a value")
        #
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addWidget(label1)
        vbox2.addWidget(self.lineedit1)
        vbox2.addWidget(label3)
        vbox2.addWidget(self.lineedit3)
        vbox2.addWidget(label2)
        vbox2.addWidget(self.lineedit2)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox2)
        self.saveButton = QtWidgets.QPushButton('Save')
        cancelButton = QtWidgets.QPushButton("Cancel")
        hbox2 = QtWidgets.QHBoxLayout()
        hbox2.addStretch(1)
        hbox2.addWidget(cancelButton)
        hbox2.addWidget(self.saveButton)
        vbox3 = QtWidgets.QVBoxLayout()
        vbox3.addLayout(hbox)
        vbox3.addLayout(hbox2)
        self.clip = QtWidgets.QApplication.clipboard()
        # connect signal for okButton
        cancelButton.clicked.connect(self.reject)
        self.setLayout(vbox3)

class AddTrainDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self,parent)
        self.setWindowTitle('Save train')
        label1 = QtWidgets.QLabel("Name or ID of the train")
        self.lineedit1 = QtWidgets.QLineEdit()
        self.lineedit1.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit1.setText("Enter a name")
        label3 = QtWidgets.QLabel("Maximum velocity of train (km/h)")
        self.lineedit3 = QtWidgets.QLineEdit()
        self.lineedit3.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit3.setText("Enter a value")
        #
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addWidget(label1)
        vbox2.addWidget(self.lineedit1)
        vbox2.addWidget(label3)
        vbox2.addWidget(self.lineedit3)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox2)
        self.saveButton = QtWidgets.QPushButton('Save')
        cancelButton = QtWidgets.QPushButton("Cancel")
        hbox2 = QtWidgets.QHBoxLayout()
        hbox2.addStretch(1)
        hbox2.addWidget(cancelButton)
        hbox2.addWidget(self.saveButton)
        vbox3 = QtWidgets.QVBoxLayout()
        vbox3.addLayout(hbox)
        vbox3.addLayout(hbox2)
        self.clip = QtWidgets.QApplication.clipboard()
        # connect signal for okButton
        cancelButton.clicked.connect(self.reject)
        self.setLayout(vbox3)
class MyGlobalResultsCanvas(FigureCanvas):
    def __init__(self,parent=None):
        self.fig = Figure(facecolor="0.94")
        self.axes1 = self.fig.add_subplot(211)
        self.axes1.set_xlabel(r'Velocity (km/h)')
        self.axes1.set_ylabel(r'Acceleration (m/s$^2$)')
        self.axes2 = self.fig.add_subplot(212)
        self.axes2.set_xlabel(r'Velocity (km/h)')
        self.axes2.set_ylabel(r'DAF ($1+\varphi$)')
        self.fig.set_tight_layout(True)
        #
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        #
        FigureCanvas.updateGeometry(self)

class MyHistoryResultsCanvas(FigureCanvas):
    def __init__(self,parent=None):
        self.fig = Figure(facecolor="0.94")
        self.axes1 = self.fig.add_subplot(211)
        self.axes1.set_xlabel(r'Time (s)')
        self.axes2 = self.fig.add_subplot(212)
        self.axes2.set_xlabel(r'Frequency (Hz)')
        self.axes2.set_ylabel(r'Amplitude')
        self.fig.set_tight_layout(True)
        #
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        #
        FigureCanvas.updateGeometry(self)

class MyModalModesCanvas(FigureCanvas):
    def __init__(self,parent=None):
        self.fig = Figure(facecolor="0.94")
        self.axes = self.fig.add_subplot(111)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['left'].set_visible(False)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.fig.set_tight_layout(True)
        #
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        #
        #
        FigureCanvas.updateGeometry(self)

class AboutDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self,parent)
        self.setWindowTitle('About')
        self.resize(450,250)
        self.icono = QtWidgets.QLabel(self)
        self.icono.setGeometry(QtCore.QRect(20, 20, 140, 140))
        self.icono.setText("")
        self.icono.setPixmap(QtGui.QPixmap(direction+"/icono.png"))
        self.icono.setScaledContents(True)
        self.line1 = QtWidgets.QLabel(self)
        self.line1.setGeometry(QtCore.QRect(180,10,140,30))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.line1.setFont(font)
        self.line1.setText("CALDINTAV")
        self.line2 = QtWidgets.QLabel(self)
        self.line2.setGeometry(QtCore.QRect(180,40,70,20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.line2.setFont(font)
        self.line2.setText("v 3.0")
        self.line2.setStyleSheet("QLabel {color: blue;}")
        self.line3 = QtWidgets.QLabel(self)
        self.line3.setGeometry(QtCore.QRect(180,60,360,90))
        self.line3.setText("Caldintav is developed by \nGroup of Computational Mechanics \n\nSchool of Civil Engineering \nTechnical University of Madrid, Spain")
        self.line4 = QtWidgets.QLabel(self)
        self.line4.setGeometry(QtCore.QRect(180,160,300,20))
        self.line4.setText('<a href="http://www.mecanica.upm.es/">http://www.mecanica.upm.es</a>')
        self.line4.setOpenExternalLinks(True)
        #
        self.license = QtWidgets.QPushButton(self)
        self.license.setGeometry(QtCore.QRect(120, 200, 100, 27))
        self.license.setText("License")
        self.close = QtWidgets.QPushButton(self)
        self.close.setGeometry(QtCore.QRect(250, 200, 100, 27))
        self.close.setText("Close")
        self.close.clicked.connect(self.reject)

class LicenseDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self,parent)
        self.setWindowTitle('Lisence')
        self.resize(450,350)
        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.textBrowser.setGeometry(QtCore.QRect(20, 10, 410, 300))
        self.textBrowser.setHtml("Copyright (c) 2017 Group of Computational Mechanics (GMC)\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the &quot;Software&quot;), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED &quot;AS IS&quot;, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE")
        self.close = QtWidgets.QPushButton(self)
        self.close.setGeometry(QtCore.QRect(180,320,100,27))
        self.close.setText("Close")
        self.close.clicked.connect(self.reject)



class InteractionTrainDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self,parent)
        self.setWindowTitle('New Train for Interaction Analysis')
        self.resize(650,590)
        self.table = QtWidgets.QTableWidget()
        self.table.resize(450,250)
        self.table.setRowCount(70)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(str("Axle Dist. (m); Axle Load (N); mw (kg); mb (kg); k1 (N/m); c1 (N.s/m);").split(";"))
        # add item for table
        for i in range(70):
            for j in range(6):
                self.table.setItem(i,j, QtWidgets.QTableWidgetItem())
        label1 = QtWidgets.QLabel("Name or ID for train:")
        self.lineedit1 = QtWidgets.QLineEdit()
        self.lineedit1.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit1.setText("Enter a name")
        label2 = QtWidgets.QLabel("File name to save:")
        self.lineedit2 = QtWidgets.QLineEdit()
        self.lineedit2.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit2.setText("Enter a file name")
        label3 = QtWidgets.QLabel("Maximum speed (km/h):")
        self.lineedit3 = QtWidgets.QLineEdit()
        self.lineedit3.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit3.setText("Enter a value")
        #
        hbox1 = QtWidgets.QHBoxLayout()
        hbox1.addWidget(label1)
        hbox1.addWidget(self.lineedit1)
        hbox1.addWidget(label3)
        hbox1.addWidget(self.lineedit3)
        # 
        hbox2 = QtWidgets.QHBoxLayout()
        spacerItem = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.clearData = QtWidgets.QPushButton("Clear data")
        self.import_csv_file = QtWidgets.QPushButton("Import from file (*.dat or *.csv) ")
        hbox2.addWidget(label2)
        hbox2.addWidget(self.lineedit2)
        hbox2.addItem(spacerItem)
        hbox2.addWidget(self.clearData)
        hbox2.addWidget(self.import_csv_file)
        #
        hbox3 = QtWidgets.QHBoxLayout()
        self.saveButton = QtWidgets.QPushButton('Save')
        cancelButton = QtWidgets.QPushButton("Cancel")
        hbox3.addStretch(1)
        hbox3.addWidget(cancelButton)
        hbox3.addWidget(self.saveButton)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.table)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        self.clip = QtWidgets.QApplication.clipboard()
        # connect signal for okButton
        cancelButton.clicked.connect(self.reject)
        self.setLayout(vbox)
        return
    def keyPressEvent(self,event):
        if (event.modifiers() & QtCore.Qt.ControlModifier):
            selected = self.table.selectedRanges()
            if  event.key() == QtCore.Qt.Key_C:  # copy
                s = ""
                for r in range(selected[0].topRow(),selected[0].bottomRow()+1):
                    for c in range(selected[0].leftColumn(),selected[0].rightColumn()+1):
                        try:
                            s += str(self.table.item(r,c).text()) + "\t"
                        except AttributeError:
                            s += "\t"
                    s = s[:-1] + "\n"
                self.clip.setText(s)
            if  event.key() == QtCore.Qt.Key_V: # paste
                try:
                    text = str(QtWidgets.QApplication.instance().clipboard().text())
                    clip_text = text.splitlines()
                    for r in range(len(clip_text)):
                        row = clip_text[r].split()
                        for c in range(len(row)):
                            self.table.setItem(r,c, QtWidgets.QTableWidgetItem(row[c]))
                except AttributeError:
                    QtWidgets.QMessageBox.question(self,'Error','Can not paste the data', QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            #  
            super(InteractionTrainDialog,self).keyPressEvent(event)
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

def  run_gui():
    # create PyQt4 application object
    app = QtWidgets.QApplication(sys.argv)
    # call the function
    myapp = caldintav()
    # show the app
    myapp.show()
    # exit
    sys.exit(app.exec_())
