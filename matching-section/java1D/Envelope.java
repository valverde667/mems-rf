/*
Java Envelope Code   ---  J. Novotny,  W. Fawley   LBNL
*/
 
/* LATEST CHANGE:
   The colors used in the plot can be changed.
   Michiel 08/15/a.
*/

import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.FlowLayout;
import java.awt.BorderLayout;
import java.awt.Graphics;
import java.awt.event.*;
import java.awt.Graphics2D;
import java.awt.print.*;
import java.awt.geom.*;
import java.awt.font.*;

import java.awt.Font;
import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.border.*;

public class Envelope extends JFrame implements ActionListener, ChangeListener, ProcessNewValues {

	public Envelope() {
		MakeGUI();
		InitParams();
	}

	private void MakeGUI () {
		setTitle("Envelope Model");

		JMenuBar mbar = new JMenuBar();
		JMenu mf = new JMenu("File");
		JMenu mo = new JMenu("Options");
		JMenu mh = new JMenu("About");

		loadValM = new JMenuItem("Load Input File");
		saveValM = new JMenuItem("Save Beam/Lattice Values => Output File");
		saveOutM = new JMenuItem("Save Envelope Output => Output File");
		exitM = new JMenuItem("Exit");
		optionsM = new JMenuItem("Printing options");
		colorM = new JMenuItem("Change colors");
		aboutM = new JMenuItem("About...");

		mf.add(loadValM);
		mf.add(saveValM);
                mf.addSeparator();
		mf.add(saveOutM);
		mf.add(exitM);
		mo.add(optionsM);
		mo.add(colorM);
		mh.add(aboutM);

		mbar.add(mf);
		mbar.add(mo);
		mbar.add(mh);
		setJMenuBar(mbar);
		DataWin = new GraphPanel();

		int buttonheight = 40;
		paramB = new JButton("Parameters");
                paramB.setBackground(Color.yellow);
                paramB.setPreferredSize (new Dimension (120,buttonheight));

		elemB = new JButton("Elements");
                elemB.setBackground(Color.green);
                elemB.setPreferredSize (new Dimension (140,buttonheight));

		initcondB = new JButton("Initial Conditions");
                initcondB.setBackground(Color.cyan);
                initcondB.setPreferredSize (new Dimension (180,buttonheight));

                matchB = new JButton("Match");
                matchB.setForeground(Color.yellow);
                matchB.setBackground(Color.blue);
                matchB.setPreferredSize (new Dimension (110,buttonheight));

		runB = new JButton("Run");
		runB.setForeground(Color.yellow);
                runB.setBackground(Color.red);
                runB.setPreferredSize (new Dimension (80,buttonheight));

		printB = new JButton("Print");
                printB.setBackground(Color.orange);
                printB.setPreferredSize (new Dimension (90,buttonheight));
		
		quadL = new JLabel();
		quadL.setToolTipText ("To select a quad for the slider, press the 'Elements' button below");
		quadValL= new JLabel();

		multB = new JButton("* 2");
                multB.setBackground(Color.cyan);
                multB.setToolTipText ("Press this button to double the scale of the slider");

		divB = new JButton("/ 2");
                divB.setBackground(Color.red);
                divB.setToolTipText ("Press this button to halve the scale of the slider");

		qvSlider = new JSlider(JSlider.HORIZONTAL);
		qvSlider.setPaintTicks (true);
		qvSlider.setPaintTrack (true);
		qvSlider.setPaintLabels (true);
		qvSlider.setBorder (new TitledBorder ("Adjust voltage"));
                int min = -100;
                int max = +100;
                int center = 0;
                int width = 1;
                qvSlider.setModel (new DefaultBoundedRangeModel (center,width,min,max));
                qvSlider.setLabelTable (qvSlider.createStandardLabels ((max-min)/4));
                qvSlider.setMajorTickSpacing ((max-min)/4);
                qvSlider.setMinorTickSpacing ((max-min)/20);
		qvSlider.addChangeListener(this);

		divB.addActionListener(this);
		multB.addActionListener(this);
		paramB.addActionListener(this);
		elemB.addActionListener(this);
		matchB.addActionListener(this);
		initcondB.addActionListener(this);
		runB.addActionListener(this);
		printB.addActionListener(this);
		loadValM.addActionListener(this);
		saveValM.addActionListener(this);
		saveOutM.addActionListener(this);
		exitM.addActionListener(this);
		optionsM.addActionListener(this);
		colorM.addActionListener(this);
		aboutM.addActionListener(this);

		JPanel p1 = new JPanel();
		p1.setPreferredSize (new Dimension (740,90));
		
		qvSlider.setPreferredSize (new Dimension (350,80));

		JPanel p1left = new JPanel();
		p1left.setLayout (new GridLayout (2,2));
		p1left.setPreferredSize (new Dimension (270,90));
		p1left.add (new JLabel ("Lattice element:"));
		p1left.add (quadL);
		p1left.add (new JLabel ("Voltage:"));
		p1left.add (quadValL);
		
		JPanel p1center = new JPanel();
		p1center.setPreferredSize (new Dimension (80,90));
		p1center.add (multB, BorderLayout.NORTH);
		p1center.add (divB, BorderLayout.SOUTH);
		
		p1.add (p1left, BorderLayout.WEST);
		p1.add (p1center, BorderLayout.CENTER);
		p1.add (qvSlider, BorderLayout.EAST);

		DataWin.setPreferredSize(new Dimension(740,470));

		JPanel p2 = new JPanel();
		p2.setPreferredSize (new Dimension (740,40));
		p2.setLayout (new FlowLayout (FlowLayout.CENTER, 0, 0));
		p2.add (paramB);
		p2.add (elemB);
		p2.add (initcondB);
		p2.add (matchB);
		p2.add (runB);
		p2.add (printB);
		
		getContentPane().setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));
		getContentPane().add (p1);
		getContentPane().add (DataWin);
		getContentPane().add (p2);

		FileDlg = new JFileChooser ();
	}
	
	private void InitParams() {	

		int NumSteps = 50;

		Emittance = 5.0e-9;		// 5e-9 m-rad		
		Current = 0.0048;		// 4.8 mA
		BeamEnergy = 160e3;		// 160kV
		Mass = 133.0;			// Cesium (u)
		beamVal = new BeamValues();
		beamVal.setBeamValues(BeamEnergy, Current, Mass, Emittance, NumSteps);

		int NumElements = 5;	
		lattice = new Element[NumElements];
		lattice[0] = new Element ("drift", "D1");
		lattice[1] = new Element ("quad", "Q1");
		lattice[2] = new Element ("drift", "D2");
		lattice[3] = new Element ("quad", "Q2");
		lattice[4] = new Element ("drift", "D3");
		lattice[0].setLength (0.015);
		lattice[1].setLength (0.05);
		lattice[2].setLength (0.12);
		lattice[3].setLength (0.1);
		lattice[4].setLength (0.0291);
		lattice[1].setVoltage (2.0e3);
		lattice[3].setVoltage (-2.0e3);

		aperture = 0.011;	// units are meters => 11 mm

		int totalNumSteps = NumElements * NumSteps;
		VecZ = new double[totalNumSteps+1];
		Data = new double[5][totalNumSteps+1];

		setSlider (0);

		setNewValues();
	}

	public void setSlider (int index) {
		int i;
		int j = 0;
		for (i = 0; i < lattice.length; i++) {
			if (lattice[i].getType().equals("quad")) {
				if (j == index) sliderindex = i;
				j++;
			}
		}
		scale = 1.0;
		slidercenter = lattice[sliderindex].getVoltage();
		qvSlider.setValue (0);
                updateSlider();
	}

	public void getDlgValues(JDialog source, Object newVal) {
		if (source instanceof ParamDialog) {
			beamVal = (BeamValues) newVal;
		} else if (source instanceof InitCondDialog) {
			envelopeVal = (double[]) newVal;
		} else if (source instanceof ElementDialog) {
			double[] newQuadVal = (double[]) newVal;
			int j = 0;
			int i;
			for (i = 0; i < lattice.length; i++) {
                               if (lattice[i].getType().equals("quad")) {
					lattice[i].setVoltage (newQuadVal[j]);
                                        j++;
                                }
			}
			updateSlider();		
			setNewValues();
		} else if (source instanceof MatchDialog) {
		// the match dialog returns the new voltages
			double[] newQuadVal = (double[]) newVal;
			int j = 0;
			int i;
			for (i = 0; i < lattice.length; i++)
                        	if (lattice[i].getType().equals("quad")) {
                                        lattice[i].setVoltage (newQuadVal[j]);
                                        j++;
                                }
			setNewValues();
		}
	}

	public void updateSlider() {
                DecimalFormat df = new DecimalFormat("0.##E0");
		quadL.setText(lattice[sliderindex].getName());
		quadValL.setText(df.format (slidercenter));
		Hashtable labelTable = new Hashtable();
		if (slidercenter > 0) scale = Math.abs (scale);
		if (slidercenter < 0) scale = - Math.abs (scale);
		int i;
		for (i = -100; i <= 100; i += 50) {
			double voltage = (1.0 + scale * i / 100.0) * slidercenter;
			String text = df.format (voltage);
			labelTable.put (new Integer(i), new JLabel (text));
		}
		qvSlider.setLabelTable(labelTable);
	}

	public void getFileValues(File inputfile) {
		int i, numsteps, numelem;
		double[] beamFileVal = new double[4];
		double[] initCond = new double[4];
		double[] finalCond = {0.0, 0.0, 0.0, 0.0};
		double[] elementlength;
		String[] elementname;
		String[] elementtype;
		double[] quadvoltage;

		try {
			FileReader fileReader = new FileReader(inputfile);
			BufferedReader dataStream = new BufferedReader(fileReader);
		
		// get BeamValue vars.
			for (i = 0; i < 4; i++) beamFileVal[i] = getNextDouble(dataStream,"=");
			
			beamVal.current = beamFileVal[0];
			beamVal.energy = beamFileVal[1];
			beamVal.mass = beamFileVal[2];
			beamVal.emittance = beamFileVal[3];

		// get initial conds. & numsteps:
			for (i = 0; i < 4; i++) envelopeVal[i] = getNextDouble(dataStream,"=");
			
		// get lattice elements
			numsteps = getNextInt(dataStream, "=");
			numelem = getNextInt(dataStream, "=");
			aperture = getNextDouble(dataStream,"=");
			quadvoltage = new double[numelem];
			elementname = new String[numelem];
			elementtype = new String[numelem];
			elementname = getNextStringArr(dataStream, "=", numelem);
			elementtype = getNextStringArr(dataStream, "=", numelem);
			elementlength = getNextArray(dataStream,"=",numelem);
			quadvoltage = getNextArray(dataStream, "=", numelem);
			dataStream.close();
			
			lattice = new Element[numelem];
			for (i = 0; i < numelem; i++) {
				lattice[i] = new Element (elementtype[i], elementname[i]);
				if (elementtype[i].equals("drift") || elementtype[i].equals("quad"))
					lattice[i].setLength (elementlength[i]);
				if (elementtype[i].equals("quad")) lattice[i].setVoltage (quadvoltage[i]);
				if (elementtype[i].equals("neutralization")) lattice[i].setNeutralization (quadvoltage[i]);
			}

                        beamVal.numsteps = numsteps;

		// set current element and quad for slider
			sliderindex = 0;
			boolean quadfound = false;
			for (i = 0; i < numelem; i++) {
				if (elementtype[i].equals("quad")) quadfound = true;
			}
			if (quadfound) {
				setSlider (0);
				updateSlider();
			}
			elemB.setEnabled (quadfound);
			qvSlider.setEnabled (quadfound);
			setNewValues();
		} catch (IOException fnfe) {
			String message = "Either the file can't be found, or it is damaged!";
			JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
		}
	}

	private int getNextInt(BufferedReader dataStream, String token) throws IOException {
		// gets the next integer in a DataInputStream
		String sval = getNext(dataStream, token);
		try {
			return Integer.parseInt (sval.trim());
		} catch (IllegalArgumentException e) {
			String message = "Please enter an integer value on line:\n" + Line;
			System.out.println (e);
			JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
		}
		return 0;
	}

	private double getNextDouble(BufferedReader dataStream, String token) throws IOException {
		// gets the next double in a DataStream
		String sval = getNext(dataStream, token);
		try {
			return Double.parseDouble (sval.trim());
		} catch (NumberFormatException e) {
			String message = "Please enter a floating point value on line:\n" + Line;
			JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
		}
		return 0.0;
	}

	private double[] getNextArray(BufferedReader dataStream, String token, int numelems)
		throws IOException {
		
		// gets the next array of double values
		int i = 0;
		double[] arr = new double[numelems];
		try {
			String arrval = getNext(dataStream, token);
			StringTokenizer t = new StringTokenizer(arrval, ",");
			for (i = 0; i < numelems; i++) arr[i] = Double.parseDouble (t.nextToken().trim());
		} catch (NumberFormatException e) {
			String message = "Please enter a floating point value on line:\n" + Line;
			JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
		} catch (NoSuchElementException e) {
			StringTokenizer t = new StringTokenizer(Line, token);
			String message1 = "Insufficient values are specified on line " + t.nextToken() + "\n";
			String message2 = "(" + i + " values found, " + numelems + " values needed).";
			JOptionPane.showMessageDialog(this, message1 + message2, "Error", JOptionPane.ERROR_MESSAGE);
		}
		return arr;
	}

	private String[] getNextStringArr(BufferedReader dataStream, String token, int numelems)
		throws IOException {
		
		// gets the next array of double values
		int i;
		String[] arr = new String[numelems];
		try {
			String arrval = getNext(dataStream, token);
			StringTokenizer t = new StringTokenizer(arrval, ",");
			for (i = 0; i < numelems; i++) {
				arr[i] = t.nextToken().trim();
			}
		} catch (NoSuchElementException e) {
			String message = "Not enough values are specified on line:\n" + Line;
			JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
		}
		return arr;
	}

	
	private String getNext(BufferedReader dataStream, String token) throws NoSuchElementException {
		try {
			Line = dataStream.readLine();
			if (Line != null) {
				StringTokenizer t = new StringTokenizer(Line, token);
				String varname = t.nextToken();	// this string has variable name
				return t.nextToken();		// contains value
			}
			else {
				String message = "Empty line found in the inputfile";
				JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
				return "";
			}
		} catch (IOException e) {
			String message = "Unable to read line:\n" + Line;
			JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
		}
		return "";
	}

	private void saveOutput(File outputfile) {
		int i;

		try {
			FileOutputStream fileStream = new FileOutputStream(outputfile);
			PrintWriter os = new PrintWriter(fileStream);

			// save object values and solution to file
			os.println("Current (A)= " + beamVal.current);
			os.println("Energy (V)= " + beamVal.energy);
			os.println("Mass (u)= " + beamVal.mass);
			os.println("Emittance (m-rad)= " + beamVal.emittance);
			os.println("");
			os.println("    Z      \t  a (m)    \t   a'    \t  b (m) \t   b'");
			os.println("");
			DecimalFormat df = new DecimalFormat("0.######E0");
			for (i = 0; i < VecZ.length; i++) {
				os.print (df.format (VecZ[i]) + "\t");
				os.print (df.format (Data[0][i]) + "\t");
                                os.print (df.format (Data[1][i]) + "\t");
                                os.print (df.format (Data[2][i]) + "\t");
                                os.print (df.format (Data[3][i]) + "\n");
			}
			os.close();
		} catch (IOException err) {
			String message = "Unable to write to the file!\n";
			JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
		}
	}

	private void saveValues (File outputfile) {
		int i;

		try {
			FileOutputStream fileStream = new FileOutputStream(outputfile);
			PrintWriter os = new PrintWriter(fileStream);

			// save object values and solution to file
			os.println("Current (A)= " + beamVal.current);
			os.println("Energy (V)= " + beamVal.energy);
			os.println("Mass (u)= " + beamVal.mass);
			os.println("Emittance (m-rad)= " + beamVal.emittance);
			os.println("a(0) = " + envelopeVal[0]);
			os.println("a'(0) = " + envelopeVal[1]);
			os.println("b(0) = " + envelopeVal[2]);
			os.println("b'(0) = " + envelopeVal[3]);
			os.println("numsteps = " + beamVal.numsteps);
			os.println("numelements = " + lattice.length);
			os.println("aperture = " + aperture);
			os.print("elementnames = ");
			for (i = 0; i < lattice.length - 1; i ++) {
				os.print(lattice[i].getName() + ", ");
			}
			os.print(lattice[i].getName());
			os.println("");
			os.print("elementtypes = ");
			for (i = 0; i < lattice.length - 1; i ++) {
				os.print(lattice[i].getType() + ", ");
			}
			os.print(lattice[i].getType());
			os.println("");
			os.print("elementlengths = ");
			DecimalFormat df = new DecimalFormat("0.######E0");
			for (i = 0; i < lattice.length; i ++) {
				String type = lattice[i].getType();
				if (type.equals("drift") || type.equals("quad")) os.print(df.format(lattice[i].getLength()));
				else if (type.equals("neutralization")) os.print ("0.0");
				else {
					System.out.println ("Program bug found: Element with type " + type);
					System.exit(0);
				}
				if (i < lattice.length - 1) os.print (", ");
				else os.println ("");
			}
			os.print("quadvoltages = ");
                        for (i = 0; i < lattice.length; i ++) {
                                String type = lattice[i].getType();
                                if (type.equals("quad")) os.print(df.format(lattice[i].getVoltage()));
                                else if (type.equals("neutralization")) os.print(df.format(lattice[i].getNeutralization()));
				else if (type.equals("drift")) os.print ("0.0");
                                else {
                                        System.out.println ("Program bug found: Element with type " + type);
                                        System.exit(0);
                                }
                                if (i < lattice.length - 1) os.print (", ");
                                else os.println ("");
                        }
			os.println ("Final z = " + VecZ[VecZ.length-1] + " m:");
			os.println("a = " + Data[0][VecZ.length-1] + " m;");
			os.println("a' = " + Data[1][VecZ.length-1] + ";");
			os.println("b = " + Data[2][VecZ.length-1] + " m;");
			os.println("b' = " + Data[3][VecZ.length-1] + ".");
			os.close();
		} catch (IOException err) {
			String message = "Unable to write to the file!";
			JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
		}
	}
	
	private void setNewValues() {
		BeamSolver newBeam = new BeamSolver(beamVal, lattice, envelopeVal, aperture);
		VecZ = newBeam.getRange();
		Data = newBeam.getData();
		DataWin.Initialize (VecZ, Data, lattice);
		DataWin.repaint();
		DecimalFormat df = new DecimalFormat("0.####E0");
		int n = VecZ.length - 1;
		System.out.print ("z = " + df.format(VecZ[n]) + " m; \t");
		System.out.print ("a = " + df.format(Data[0][n]) + " m; \t");
                System.out.print ("a' = " + df.format(Data[1][n]) + " rad; \t");
                System.out.print ("b = " + df.format(Data[2][n]) + " m; \t");
                System.out.print ("b' = " + df.format(Data[3][n]) + " rad.\n");

	}

	public void stateChanged(ChangeEvent evt) {  
		double newvalue = (1.0 + scale*qvSlider.getValue()/100.0) * slidercenter;
                DecimalFormat df = new DecimalFormat("0.###E0");
		quadValL.setText (df.format(newvalue));
		lattice[sliderindex].setVoltage (newvalue);
		setNewValues();
	 }
	
	public void actionPerformed(ActionEvent evt) {
		// check all button options
		Object source = evt.getSource();
		if (source == divB) {
			scale /= 2.0;
			updateSlider();
		} else if (source == multB) {
			scale *= 2.0;
			updateSlider();
		} else if (source == paramB) {
			ParamDialog ParamDlg = new ParamDialog(this, beamVal);
			ParamDlg.show();
		} else if (source == elemB) {
			ElementDialog ElementDlg = new ElementDialog(this, lattice);
			ElementDlg.show();
		} else if (source == matchB) {
			MatchDialog MatchDlg = new MatchDialog(this, beamVal, lattice, envelopeVal, aperture);
			MatchDlg.show();
		} else if (source == initcondB) {
			InitCondDialog InitCondDlg = new InitCondDialog(this, envelopeVal);
			InitCondDlg.show();
		} else if (source == saveOutM) {
			int returnVal = FileDlg.showSaveDialog (this);
			if (returnVal == JFileChooser.APPROVE_OPTION) saveOutput(FileDlg.getSelectedFile());
		} else if (source == saveValM) {
			int returnVal = FileDlg.showSaveDialog (this);
			if (returnVal == JFileChooser.APPROVE_OPTION) saveValues(FileDlg.getSelectedFile());
		} else if (source == loadValM) {
			int returnVal = FileDlg.showOpenDialog (this);
			if (returnVal == JFileChooser.APPROVE_OPTION) getFileValues(FileDlg.getSelectedFile());
		} else if (source == optionsM) DataWin.showOptionsWindow();
		  else if (source == colorM) DataWin.showColorChooser();
		  else if (source == aboutM) {
			String message1 = "K-V Beam Envelope Solver\n";
			String message2 = "Jason Novotny/William Fawley LBNL;\n";
			String message3 = "Modified by Michiel de Hoon 08/15/2000.";
			String message = message1 + message2 + message3;
			JOptionPane.showMessageDialog(this,message, "About...", JOptionPane.INFORMATION_MESSAGE);
		} else if (source == runB) setNewValues();
		  else if (source == printB) {
			PrinterJob printJob = PrinterJob.getPrinterJob();
			PageFormat pf = printJob.defaultPage();
			pf = printJob.pageDialog (pf);
			Paper paper = pf.getPaper();

			double margin = 72; // 72 points = 1 inch margin
			double x = margin;
			double y = margin;
			double width = paper.getWidth() - 2 * margin;
			double height = paper.getHeight() - 2 * margin;
			paper.setImageableArea(x,y,width,height);
			pf.setPaper (paper);
			printJob.setPrintable(DataWin, pf);
			if (printJob.printDialog()) {
				try {
					printJob.print();
				} catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		} else if (source == exitM) {
			System.exit(0);
		} 
	}

	public static void main(String[] args) {  
		JFrame f = new Envelope();
		f.setSize(740,660);
		f.show();
	}

	private BeamValues beamVal;
	private double[] envelopeVal = {0.0035, -0.005, 0.0035, -0.005}; // a(0), a'(0), b(0), b'(0);
	private Element[] lattice;
	private GraphPanel DataWin;
	private JSlider qvSlider;
	private int sliderindex;
	private JLabel quadL, quadValL;
	private JButton multB, divB, paramB, elemB, initcondB, matchB, printB, runB;
	private JMenuItem loadValM, saveValM, saveOutM, exitM, optionsM, colorM, aboutM;
	private JFileChooser FileDlg;
	private double[][] Data;
	private double[] VecZ;
	
	private double PerveanceQ, Emittance, Current, Mass, BeamEnergy, aperture;
	private String Line;
	private double scale;
	private double slidercenter;

}  //  End Class Envelope

class Element {

	public Element() {}

	public Element (String type, String name) {
		if (type.equals ("drift") || type.equals("quad") || type.equals("neutralization")) this.type = type;
		else {
			System.out.println ("Error while reading input file");
			System.out.println ("Element type was neither drift nor quad nor neutralization");
			System.out.println ("Type read was " + type);
                        System.exit(0);
		}
		this.name = name;
		this.length = length;
		this.voltage = 0.0;
		this.neutralization = 0.0;
	}

	public void setLength (double length) {
		if (length >= 0) this.length = length;
		else {
			System.out.println ("Program bug found: Trying to set the length of a lattice element to a negative value");
			System.exit (1);
		}
	}
		
	public void setVoltage (double voltage) {
		if (type.equals("quad")) this.voltage = voltage;
		else {
			System.out.println ("Program bug found: Trying to put a voltage on an element that is not a quad");
			System.exit (1);
		}
	}

	public void setNeutralization (double neutralization) {
		if (type.equals("neutralization")) this.neutralization = neutralization;
                else {
			System.out.println ("Program bug found: Trying to put a neutralization value on a " + type);
			System.exit (1);
		}
        }

	public String getName() {
		return name;
	}

	public String getType() {
		return type;
	}

        public double getLength () {
                if (type.equals("quad") || type.equals("drift")) return length;
                else {
			System.out.println ("Program bug found: Trying to read the length of an element that is neither a drift nor a quad");
			System.exit (1);
			return 0.0;
		}
        }

        public double getVoltage () {
                if (type.equals("quad")) return voltage;
                else {
			System.out.println ("Program bug found: Trying to read the voltage on an element that is not a quad");
			System.exit (1);
			return 0.0;
		}
        }

        public double getNeutralization () {
                if (type.equals("neutralization")) return neutralization;
                else {
                        System.out.println ("Program bug found: Trying to read the neutralization of a " + type);
                        System.exit (1);
                        return 0.0;
                }
        }

	private String type;
	private String name;
	private double length;
        private double voltage;
	private double neutralization;
}

class BeamValues {
	
	public BeamValues() {}
		
	public void setBeamValues(double eng, double curr, double m, double emit, int nsteps) {
		energy = eng;
		current = curr;
		mass = m;
		emittance = emit;
		numsteps = nsteps;
	}

	double energy, current, mass, emittance;
	int numsteps;
	
}

interface ProcessNewValues {
	public void setSlider (int index);
	public void getDlgValues(JDialog source, Object obj);
}


class ParamDialog extends JDialog implements ActionListener {
	
	public ParamDialog(JFrame parent, BeamValues beam) {
		super(parent, "Beam Parameters", false);
		DecimalFormat df = new DecimalFormat("0.######E0");

		JPanel p1 = new JPanel();
		p1.setPreferredSize (new Dimension (280,250));
		p1.setLayout(new GridLayout(5,2));
		p1.add(new JLabel("Current (A):"));
		p1.add(currentTF = new JTextField(df.format(beam.current), 6));
		p1.add(new JLabel("Energy (V):"));
		p1.add(energyTF = new JTextField(df.format(beam.energy), 6));
		p1.add(new JLabel("Mass (u):"));
		p1.add(massTF = new JTextField(df.format(beam.mass), 6));
		p1.add(new JLabel("Emittance (m-rad):"));
		p1.add(emittanceTF = new JTextField(df.format(beam.emittance), 6));
		p1.add(new JLabel("Number of steps per element:"));
		p1.add(numStepsTF = new JTextField("" + beam.numsteps));
		getContentPane().add("Center", p1);

		JPanel p2 = new JPanel();
		okB = new JButton("Ok");
                okB.setBackground(Color.green);
                okB.setSize(60,40);

		cancelB = new JButton("Cancel");
                cancelB.setBackground(Color.red);

		okB.addActionListener(this);
		cancelB.addActionListener(this);
		p2.add(okB);
		p2.add(cancelB);
		getContentPane().add("South", p2);
		setSize(480,250);
	}

	public void actionPerformed(ActionEvent evt) {
		Object source = evt.getSource();
		if (source == okB) {
			dispose();
			BeamValues newBeamVal = new BeamValues();
			try {
                	        newBeamVal.current = Double.parseDouble (currentTF.getText());
				newBeamVal.energy = Double.parseDouble (energyTF.getText());
				newBeamVal.mass = Double.parseDouble (massTF.getText());
				newBeamVal.emittance = Double.parseDouble (emittanceTF.getText());
				newBeamVal.numsteps = Integer.parseInt (numStepsTF.getText());
			} catch(NumberFormatException e) {
				String message = "Please enter a numerical value";
				JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
			}
			((ProcessNewValues)getParent()).getDlgValues(this, newBeamVal);
		} else if (source == cancelB) {
			dispose();
		} 
	}

	private JButton okB, cancelB;
	private JTextField currentTF, energyTF, massTF, emittanceTF, numStepsTF;
}

class ElementDialog extends JDialog implements ActionListener, ListSelectionListener {

	public ElementDialog(JFrame parent, Element[] lattice) {
		super(parent, "Lattice Elements", false);

		// First count the number of quads
		int numQuads = 0;
		int i;
                for (i = 0; i < lattice.length; i++) {
                        if (lattice[i].getType().equals("quad")) numQuads++;
		}

		name = new String[numQuads];
		voltage = new double[numQuads];
		listModel = new DefaultListModel();
		
		int j = 0;	
		for (i = 0; i < lattice.length; i++) {
			if (lattice[i].getType().equals("quad")) {
				name[j] = lattice[i].getName();
				voltage[j] = lattice[i].getVoltage();
				listModel.addElement ("Element: " + name[j] + "              Voltage: " + voltage[j]);
				j = j + 1;
			}
		}

		int whichquad = 0;

		quadList = new JList(listModel);
                quadList.setBackground(Color.yellow);
		quadList.setSelectedIndex (whichquad);
		
		JScrollPane listScrollPane = new JScrollPane(quadList);
		int quadListSize = 300;
		listScrollPane.setPreferredSize (new Dimension (320,quadListSize));
	
		sliderB = new JButton("Slider");
                sliderB.setBackground(Color.blue);
                sliderB.setForeground(Color.orange);

		updateB = new JButton("Update");
                updateB.setBackground(Color.green);

		elemL = new JLabel(name[whichquad] + ": ");
		JLabel quadL = new JLabel("Voltage: ");
		quadTF = new JTextField("" + voltage[whichquad], 8);
		quadTF.setToolTipText ("Hit return after entering a numerical value here");
		
		sliderB.addActionListener(this);
		updateB.addActionListener(this);
		quadList.addListSelectionListener(this);
		quadTF.addActionListener(this);

		JPanel p1 = new JPanel();
		p1.setLayout(new FlowLayout());
		p1.add(elemL);
		p1.add(quadL);
		p1.add(quadTF);
		JPanel p2 = new JPanel();
		p2.setLayout(new FlowLayout());
		p2.add(sliderB);
		p2.add(updateB);

		getContentPane().add("North", listScrollPane);
		getContentPane().add("Center", p1);
		getContentPane().add("South", p2);
		
		setSize(350,150+quadListSize);
	}
	
	public void actionPerformed(ActionEvent evt) {
		Object source = evt.getSource();
		int whichquad = quadList.getSelectedIndex();
		if (source == sliderB) {
			dispose();
			whichquad = quadList.getSelectedIndex();
			((ProcessNewValues) getParent()).setSlider (whichquad);
		} else if (source == updateB) {
		   dispose();
		   ((ProcessNewValues)getParent()).getDlgValues(this, voltage);
		}  else if (source == quadTF) {
			try {
				String newQuadS = quadTF.getText();
				listModel.setElementAt ("Element: " + elemL.getText() + "              Voltage: " + newQuadS, whichquad);
				voltage[whichquad] = Double.parseDouble (newQuadS);
			} catch(NumberFormatException e) {
				String message = "Please enter a floating point value";
				JOptionPane.showMessageDialog(this, message, "Error", JOptionPane.WARNING_MESSAGE);
			}
		}
	}

	public void valueChanged(ListSelectionEvent evt) {
		if (evt.getValueIsAdjusting() == false) {
			int whichquad = quadList.getSelectedIndex();
			elemL.setText(name[whichquad]);
			quadTF.setText("" + voltage[whichquad]);
		}
	}
	
	private String[] name;
	private double[] voltage;
	private JButton sliderB, updateB;
	private JTextField quadTF;
	private JLabel elemL;
	private JList quadList;
	private DefaultListModel listModel;
}

class MatchDialog extends JDialog implements ActionListener {
	public MatchDialog(JFrame parent, BeamValues beam, Element[] lattice, double[] envelopeVal, double aperture) {
		super(parent, "Match", false);
		
                // First count the number of quads
                numQuads = 0;
                int i;
                for (i = 0; i < lattice.length; i++) {
                        if (lattice[i].getType().equals("quad")) numQuads++;
                }

		initialQuadVoltageTextField = new JTextField[numQuads];
		finalQuadVoltageTextField = new JTextField[numQuads];
		indices = new int[4];
		envelopeTextField = new JTextField[4];
		varyQuadCheckBox = new JCheckBox[numQuads];
		localBeamSolver = new BeamSolver(beam, lattice, envelopeVal, aperture);
		finalEnvelopeValues = localBeamSolver.getFinalData();
		quadVoltage = new double[numQuads];
		
		JPanel p1 = new JPanel();
                p1.setLayout(new GridLayout(numQuads+1,4,1,1));
                p1.add (new JLabel ("Element", JLabel.CENTER));
                p1.add (new JLabel ("Initial voltage (V)", JLabel.CENTER));
                p1.add (new JLabel ("Final voltage (V)", JLabel.CENTER));
                p1.add (new JLabel ("Vary?", JLabel.LEFT));
		
		int j = 0;
		DecimalFormat df = new DecimalFormat("0.######E0");
                for (i = 0; i < lattice.length; i++) {
			if (lattice[i].getType().equals("quad")) {
				String text = df.format (lattice[i].getVoltage());
				initialQuadVoltageTextField[j] = new JTextField (text);
				finalQuadVoltageTextField[j] = new JTextField (text);
				finalQuadVoltageTextField[j].setEditable (false);
	                        initialQuadVoltageTextField[j].addActionListener(this);
	                        varyQuadCheckBox[j] = new JCheckBox ();
       	                	p1.add (new JLabel (lattice[i].getName(), JLabel.CENTER));
	                        p1.add (initialQuadVoltageTextField[j]);
				p1.add (finalQuadVoltageTextField[j]);
				p1.add (varyQuadCheckBox[j]);
				j++;
			}
                }
		
		JPanel p2 = new JPanel();
		p2.setSize (200,200);
		p2.setLayout(new GridLayout(5,2,1,1));
		for (i = 0; i < 4; i++) envelopeTextField[i] = new JTextField (df.format(finalEnvelopeValues[i]));
		p2.add (new JLabel ());
		p2.add (new JLabel ());
		p2.add (new JLabel ("a (m)", JLabel.CENTER));
		p2.add (envelopeTextField[0]);
		p2.add (new JLabel ("a' (rad)", JLabel.CENTER));
		p2.add (envelopeTextField[1]);
                p2.add (new JLabel ("b (m)", JLabel.CENTER));
		p2.add (envelopeTextField[2]);
                p2.add (new JLabel ("b' (rad)", JLabel.CENTER));
		p2.add (envelopeTextField[3]);
		
                cancelB = new JButton("Cancel");
                cancelB.setBackground(Color.blue);
                cancelB.setForeground(Color.yellow);
		
		applyB = new JButton("Apply");
                applyB.setBackground(Color.green);
		applyB.setEnabled(false);
                
                startB = new JButton("Start");
                startB.setBackground(Color.yellow);
                
		stepB = new JButton("Step");
                stepB.setBackground(Color.red);
                stepB.setForeground(Color.yellow);
		stepB.setEnabled(false);

		manyStepB = new JButton ("20 steps");
		manyStepB.setBackground(Color.black);
		manyStepB.setForeground(Color.yellow);
		manyStepB.setEnabled(false);

		continueB = new JButton ("Continue");
		continueB.setBackground (Color.cyan);
                
                cancelB.addActionListener(this);
		applyB.addActionListener(this);
		startB.addActionListener(this);
		stepB.addActionListener(this);
		manyStepB.addActionListener(this);
		continueB.addActionListener(this);
		
		JPanel p3 = new JPanel();
		p3.setLayout(new FlowLayout());
		p3.add(cancelB);
		p3.add(applyB);
		p3.add(startB);
		p3.add(stepB);
		p3.add(manyStepB);
		p3.add(continueB);

		getContentPane().add("North", p1);
		getContentPane().add("Center", p2);
		getContentPane().add("South", p3);
		
		setSize(600,270 + 30 * numQuads);
	}
	
	public void actionPerformed(ActionEvent evt) {
		Object source = evt.getSource();
		DecimalFormat df = new DecimalFormat("0.######E0");
		if (source == applyB) {
			((ProcessNewValues)getParent()).getDlgValues (this, quadVoltage);
			dispose();
		} else if (source == startB) {
			if (Initialise()) {
				stepB.setEnabled(true);
				manyStepB.setEnabled(true);
			}
		} else if (source == stepB) {
			if (DoIterationStep ()) {
				int i;
				for (i = 0; i < 4; i++) {
					int j = indices[i];
					finalQuadVoltageTextField[j].setText (df.format(quadVoltage[j]));
				}
				System.out.println ("***** Iteration finished *****");
				stepB.setEnabled(false);
				manyStepB.setEnabled(false);
				applyB.setEnabled(true);
			}
		} else  if (source == manyStepB) {
			int counter = 0;
			while (counter < 20) {
	                        if (DoIterationStep ()) {
                                	int i;
	                                for (i = 0; i < 4; i++) {
       		                                int j = indices[i];
                                        	finalQuadVoltageTextField[j].setText (df.format(quadVoltage[j]));
                                	}
                                	System.out.println ("***** Iteration finished *****");
                                	stepB.setEnabled(false);
					manyStepB.setEnabled(false);
                                	applyB.setEnabled(true);    
					counter = 20;
				}
				else counter++;
                        }
		} else if (source == continueB) {
			int i;
			for (i = 0; i < numQuads; i++) initialQuadVoltageTextField[i].setText (finalQuadVoltageTextField[i].getText());
		} else if (source == cancelB) {
			dispose();				
		}
	}
	
	private boolean Initialise () {
		final double STPMX = 100.0;
		int i;
		int j = 0;
		for (i = 0; i < numQuads; i++) {
			if (varyQuadCheckBox[i].isSelected()) {
				if (j < 4) indices[j] = i;
				j++;
			}
		}
		if (j != 4) {
			System.out.println ("Exactly four quads should be selected! " + j + " quads were selected.");
			return false;
		}
		for (i = 0; i < numQuads; i++) quadVoltage[i] = Double.parseDouble (initialQuadVoltageTextField[i].getText());
		for (i = 0; i < 4; i++) finalEnvelopeValues[i] = Double.parseDouble (envelopeTextField[i].getText());
		System.out.println ("***** Starting the iteration *****");
		applyB.setEnabled(false);
		double sum = 0.0;
		localBeamSolver.setVoltages (quadVoltage);
               	localBeamSolver.Solve();
               	fvec = localBeamSolver.getFinalData();
               	for (i = 0; i < 4; i++) {
			fvec[i] -= finalEnvelopeValues[i];
			sum += fvec[i]*fvec[i];
		}
               	f = 0.5 * sum;
               	double test = 0.0;
               	for (i = 0; i < 4; i++) {
                	if (Math.abs (fvec[i]) > test) test = Math.abs (fvec[i]);
		}
               	System.out.println ("Initial error is " + test);

               	sum = 0.0;
               	for (i = 0; i < 4; i++) sum += Math.pow(quadVoltage[indices[i]],2);
               	stpmax = STPMX * Math.max (Math.sqrt(sum),4.0);
               	return true;
	}
	
	private boolean DoIterationStep () {
		final double TOLF = 1.0e-8;
		final double TOLX = 1.0e-8; // Minimum relative change in quadVoltage for the iteration to continue
		final double ALF = 1.0e-8; // Ensures sufficient decrease in function value
		final double EPS = 1.0e-8; // Approximate square root of the machine precision.
		final int n = 4; // Four voltages to be changed

                double[] xold = new double[n];
                int i;
		double[] g = new double[n];
		
		double[] p = new double[n];
                double[][] fjac = new double[n][n];
	        for (i = 0; i < n; i++) {
			double temp = quadVoltage[indices[i]];
			double h = EPS * Math.abs(temp);
			if (h == 0.0) h = EPS;
			quadVoltage[indices[i]] = temp + h; // Trick to reduce finite precision error.
			h = quadVoltage[indices[i]] - temp;
			localBeamSolver.setVoltages (quadVoltage);
			localBeamSolver.Solve();
			double[] fnew = localBeamSolver.getFinalData();
			int j;
			for (j = 0; j < n; j++) fnew[j] -= finalEnvelopeValues[j];
			quadVoltage[indices[i]] = temp;
			double sum = 0.0;
			for (j = 0; j < n; j++) {
				fjac[j][i] = (fnew[j]-fvec[j]) / h;   // Forward difference formula.
                               	sum += fjac[j][i] * fvec[j];
			}
                        g[i] = sum;
		}
		for (i = 0; i < n; i++) xold[i] = quadVoltage[indices[i]];
		for (i = 0; i < n; i++) p[i] = - fvec[i];
		
               	LUSolve (fjac, p);

		// Starting linesearch
		double fold = f;
		double sum = 0.0;
		for (i = 0; i < n; i++) sum += p[i] * p[i];
		sum = Math.sqrt(sum);
		if (sum>stpmax) for (i=0; i<n; i++) p[i] *= stpmax / sum; // Scale if attempted step is too big.

		double slope = 0.0;
		for (i = 0; i < n; i++) slope += g[i] * p[i];
		if (slope >= 0.0) {
			System.out.println ("Roundoff problem in lnsrch.");
			return false;
		}
		double test = 0.0; // Compute lambda_min
		for (i = 0; i < n; i++) {
			double temp = Math.abs (p[i]) / Math.max (Math.abs(xold[i]),1.0);
			if (temp > test) test = temp;
		}
		double alamin = TOLX / test;
		double alam = 1.0; // Always try full Newton step first.
		double f2 = 0;
		double alam2 = 0;
		boolean completed = false;
		while (! completed) {         // Start of iteration loop.
			double tmplam = 0;
			for (i = 0; i < n; i++) quadVoltage[indices[i]] = xold[i] + alam * p[i];
			localBeamSolver.setVoltages (quadVoltage);
			localBeamSolver.Solve();
			fvec = localBeamSolver.getFinalData();
			sum = 0.0;
			for (i = 0; i < n; i++) {
				fvec[i] -= finalEnvelopeValues[i];
				sum += fvec[i]*fvec[i];
			}
			f = 0.5 * sum;
			if (alam < alamin) completed = true;    // Convergence on delta x.
			else if (f <= fold + ALF * alam * slope) completed = true; // Sufficient function decrease
			else if (alam == 1.0) tmplam = - slope / (2.0 * (f-fold-slope)); // Backtrack first time
			else {                                                 // Subsequent backtracks.
				double rhs1 = f - fold - alam * slope;
				double rhs2 = f2 - fold - alam2 * slope;
				double a = (rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2);
				double b = (-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2);
				if (a == 0.0) tmplam = - slope / (2.0 * b);
				else {
					double disc = b*b-3.0*a*slope;
					if (disc < 0.0) tmplam = 0.5 * alam;
					else if (b <= 0.0) tmplam = (-b + Math.sqrt(disc))/(3.0*a);
					else tmplam = - slope / (b + Math.sqrt(disc));
				}
				if (tmplam > 0.5 * alam) tmplam = 0.5 * alam;     // lambda <= 0.5 lambda1
			}
			alam2 = alam;
            System.out.println ("alam is " + alam + " "+ tmplam); // p seidl
            System.out.println ("Quad voltages " + quadVoltage[0] + " "+ quadVoltage[1] + " "+ quadVoltage[2] + " "+ quadVoltage[3] + " "+ quadVoltage[4] + " "+ quadVoltage[5]); // p seidl
			f2 = f;
			alam = Math.max (tmplam,0.1*alam);    // lambda >= 0.1 lambda1
		}                                    // Try again
		// End of linesearch
                test = 0.0;
		for (i = 0; i < n; i++) {
			if (Math.abs(fvec[i]) > test) test = Math.abs(fvec[i]);
		}
		System.out.println ("Error is " + test);
		return (test < TOLF);
	}

	private void LUSolve (double[][] matrix, double[] solution) {
		int n = solution.length; 
		int[] index = new int[n];
		final double TINY = 1.0e-20;
		int i, j, k;
		int imax = 0; // Initialise to avoid error messages when compiling
		double[] vv = new double[n];
		for (i = 0; i < n; i++) {
			double big = 0.0;
			for (j = 0; j < n; j++) {
				double temp = Math.abs(matrix[i][j]);
				if (temp > big) big = temp;
			}
			if (big == 0.0) {   // No nonzero largest element
				System.out.println ("Singular matrix in the LU decomposition");
				return;
			}
			vv[i] = 1.0 / big;     // Save the scaling
		}
		for (j = 0; j < n; j++) {
		// This is the loop over columns of Crout's method
			for (i = 0; i < j; i++) {
				double sum = matrix[i][j];
				for (k = 0; k < i; k++) sum -= matrix[i][k] * matrix[k][j];
				matrix[i][j] = sum;
			}
			double big = 0.0;  // Initialise for the search for largest pivot element
			imax = j;   // Initialise to avoid error messages when compiling
			for (i = j; i < n; i++) {
				double sum = matrix[i][j];
				for (k = 0; k < j; k++) sum -= matrix[i][k] * matrix[k][j];
				matrix[i][j] = sum;
				double dum = vv[i] * Math.abs (sum);
				if (dum >= big) {  // Is the figure of merit better than the best so far?
					big = dum;
					imax = i;
				}
			}
			if (j != imax) {                     // Do we need to interchange rows?
				for (k = 0; k < n; k++) {      // Yes, do so ...
					double dum = matrix[imax][k];
					matrix[imax][k] = matrix[j][k];
					matrix[j][k] = dum;
				}
				vv[imax] = vv[j];                 // and also interchange the scale factor
			}
			index[j] = imax;
			if (matrix[j][j] == 0.0) matrix[j][j] = TINY;
			// If the pivot element is zero the matrix is singular (at least to the precision
			// of the algorithm). For some applications on singular matrices, it is
			// desirable to substitute TINY for zero.
			if (j + 1 != n) {
				double dum = 1.0 / matrix[j][j];
				for (i = j + 1; i < n; i++)
				matrix[i][j] *= dum;
			}
		}
		int ii = -1;
		for (i = 0; i < n; i++) {
			int ip = index[i];
			double sum = solution[ip];
			solution[ip] = solution[i];
			if (ii+1 > 0) {
				for (j = ii; j < i; j++) sum -= matrix[i][j] * solution[j];
			}
			else if (sum != 0.0) ii = i;
			solution[i] = sum;
		}
		for (i = n; i > 0; i--) {
			double sum = solution[i-1];
			for (j = i; j < n; j++) sum -= matrix[i-1][j] * solution[j];
			solution[i-1] = sum / matrix[i-1][i-1];
		}
	}

	
	private int numQuads;
	private BeamSolver localBeamSolver;
	private JTextField[] initialQuadVoltageTextField, finalQuadVoltageTextField, envelopeTextField;
	private JCheckBox[] varyQuadCheckBox;
	private JButton cancelB, applyB, startB, stepB, manyStepB, continueB;
	private double[] finalEnvelopeValues = new double[4];
	private int[] indices;
	private double stpmax;
	private double[] quadVoltage;
	private double fvec[] = new double[4];
	private double f;
}

class InitCondDialog extends JDialog implements ActionListener {
	
	public InitCondDialog(JFrame parent, double[] env) {
		super(parent, "Initial Conditions", false);
		DecimalFormat df = new DecimalFormat("0.######E0");
		JPanel p1 = new JPanel();
		p1.setLayout(new GridLayout(5,2));
		p1.add(new JLabel("a(0) (m):"));
		p1.add(aTF = new JTextField(df.format(env[0]), 6));
		p1.add(new JLabel("a'(0) (rad):"));
		p1.add(apTF = new JTextField(df.format(env[1]), 6));
		p1.add(new JLabel("b(0) (m):"));
		p1.add(bTF = new JTextField(df.format(env[2]), 6));
		p1.add(new JLabel("b'(0) (rad):"));
		p1.add(bpTF = new JTextField(df.format(env[3]), 6));
		getContentPane().add("Center", p1);
		JPanel p2 = new JPanel();
		okB = new JButton("Ok");
                okB.setBackground(Color.green);
                okB.setSize(40,25);

		cancelB = new JButton("Cancel");
                cancelB.setBackground(Color.red);

		okB.addActionListener(this);
		cancelB.addActionListener(this);
		p2.add(okB);
		p2.add(cancelB);
		getContentPane().add("South", p2);
		setSize(220,210);
	}

	public void actionPerformed(ActionEvent evt) {
		Object source = evt.getSource();
		if (source == okB) {
			dispose();
			double[] newEnvVal = new double[4];
			try {
                	        newEnvVal[0] = Double.parseDouble (aTF.getText());
				newEnvVal[1] = Double.parseDouble (apTF.getText());
				newEnvVal[2] = Double.parseDouble (bTF.getText());
				newEnvVal[3] = Double.parseDouble (bpTF.getText());
			} catch(NumberFormatException e) {
				String message = "Please enter a numerical value";
				JOptionPane.showMessageDialog(this,message, "Error", JOptionPane.ERROR_MESSAGE);
			}
			((ProcessNewValues)getParent()).getDlgValues(this, newEnvVal);
		} else if (source == cancelB) {
			dispose();
		} 
	}
	
	private JButton okB, cancelB;
	private JTextField aTF, apTF, bTF, bpTF;

}  //  END Class InitCondDialog

class BeamSolver {
	
	// The following physical constants were taken from the NIST homepage
	// at http://physics.nist.gov/cuu/Constants/index.html?/codata86.html

	public static final double C = 2.99792458e8;	// speed of light
	public static final double eps0 = 8.854187817e-12;	// permittivity of vacuum
	public static final double K = 1.0 / (4.0 * Math.PI * eps0);		// Coulomb constant = 1/4pieO
	public static final double COULOMB = 1.6021764620e-19; // electron charge
	public static final double AMU = 1.66053873e-27;		// amu in SI

	public BeamSolver(BeamValues beam, Element[] lattice, double[] envelope, double aperture) {

		int i,j;
                // First count the number of quads
                int numQuads = 0;
                for (i = 0; i < lattice.length; i++) {
                        if (lattice[i].getType().equals("quad")) numQuads++;
                }


	// create storage for output data vectors
		quadStrength = new double[lattice.length];
		stepsize = new double[lattice.length];
		quadindices = new int[numQuads];

	// save emittance value	
		emittance = beam.emittance;

	// save aperture
		this.aperture = aperture;
		
	// save mass
		mass = beam.mass;
		
	// save energy
		energy = beam.energy;
		
	// save number of steps
		numsteps = beam.numsteps;

	// save initial conditions
		for (i = 0; i < 4; i ++) initCond[i] = envelope[i];

		// calculate perveance
		double gamma = 1 + energy*COULOMB/(mass*AMU*C*C);
		double beta = Math.sqrt (Math.pow(energy*COULOMB,2.)+2*energy*COULOMB*mass*AMU*C*C) / (energy*COULOMB + mass*AMU*C*C);
		initialperveance = 2*COULOMB*K*beam.current/(AMU*beam.mass*Math.pow(beta*gamma*C,3.0));
		double totalenergy = mass * AMU * C * C + COULOMB * energy;
		
		// calculate quadrupole field strength
		j = 0;
		for (i = 0; i < lattice.length; i++) {
			if (lattice[i].getType().equals("quad")) {
				quadindices[j] = i;
				quadStrength[i] = lattice[i].getVoltage()*2.*COULOMB / (totalenergy * Math.pow (beta * aperture, 2));
				j++;
			}
			else quadStrength[i] = 0.0;
		}

		int totalNumSteps = lattice.length * beam.numsteps;
		neutralizationindex = lattice.length;
		for (i = 0; i < lattice.length; i++) {
			if (lattice[i].getType().equals("neutralization")) {
				neutralization = lattice[i].getNeutralization();
				neutralizationindex = i;
				totalNumSteps = (lattice.length - 1) * beam.numsteps;
			}
		}

		// make Z vector
		// (number of steps is constant, but element length varies, so stepsize varies per element).
		Data = new double[4][totalNumSteps+1];
		VecZ = new double[totalNumSteps+1];
		j = 0;
		VecZ[j] = 0;
		for (i = 0; i < lattice.length; i++) {
			if (i != neutralizationindex) {
				stepsize[i] = lattice[i].getLength() / beam.numsteps;
				int counter;
				for (counter = 0; counter < beam.numsteps; counter++) {
					j = j + 1;
					VecZ[j] = VecZ[j-1] + stepsize[i];
				}
			}
		}
		Solve();
	}
	
	public void setVoltages (double[] voltages) {
		int i;
                double gamma = 1 + energy*COULOMB/(mass*AMU*C*C);
		double beta = Math.sqrt (Math.pow(energy*COULOMB,2.)+2*energy*COULOMB*mass*AMU*C*C) / (energy*COULOMB + mass*AMU*C*C);
		// Relativistically correct.
		double totalenergy = mass * AMU * C * C + COULOMB * energy;
		// Relativistically correct
		for (i = 0; i < voltages.length; i++)
			quadStrength[quadindices[i]] = voltages[i] * 2.*COULOMB / (totalenergy * Math.pow (beta * aperture, 2));
	}

	public void Solve() {
		// solve for each element
		int start = 0;
		int end = 0;
		
		Data[0][start] = initCond[0];
                Data[1][start] = initCond[1];
		Data[2][start] = initCond[2];
                Data[3][start] = initCond[3];
                
		start += 1;
		int i;
		perveance = initialperveance;
		for (i = 0; i < quadStrength.length; i++) {
			if (i == neutralizationindex) perveance *= neutralization;
			else {
				end = start + numsteps - 1;
				RungeKutta(start, end, stepsize[i], i);
				start = end + 1;
			}
		}

	//  restore initial conditions again
		for (i = 0; i < 4; i ++) initCond[i] = Data[i][0];
	}

	private void RungeKutta(int start, int end, double h, int numelem) {
		// Use 4th order Runge-Kutta method to solve element
		// with a given quadrupole voltage
		int i,j;
		double acritmin, acritmax, bcritmin, bcritmax, dum;
		double[] w = new double[4];
		double[] tempw = new double[4];
		double[] k0 = new double[4];
		double[] k1 = new double[4];
		double[] k2 = new double[4];
		double[] k3 = new double[4];

		// set w to the initial conditions
		for (i = 0; i < 4; i++) w[i] = initCond[i];

		for (i = start; i <= end; i++) {
			// make quad strength vector
			
                        acritmin = -0.5*w[0]/h;
                        acritmax =  0.5*w[0]/h;
                        dum=0.2*aperture/h;
                        if (dum > acritmax) acritmax=dum;

                        bcritmin = -0.5*w[2]/h;
                        bcritmax =  0.5*w[2]/h;
                        if (dum > bcritmax) bcritmax=dum;

			for (j = 0; j < 4; j++) tempw[j] = w[j];

			k0[0] = h*Function1(tempw, acritmin, acritmax);
			k0[1] = h*Function2(numelem, tempw);
			k0[2] = h*Function3(tempw, bcritmin, bcritmax);
			k0[3] = h*Function4(numelem, tempw);
			
			for (j = 0; j < 4; j++) tempw[j] = w[j] + 0.5*k0[j];

			k1[0] = h*Function1(tempw, acritmin, acritmax);
			k1[1] = h*Function2(numelem, tempw);
			k1[2] = h*Function3(tempw, bcritmin, bcritmax);
			k1[3] = h*Function4(numelem, tempw);

			for (j = 0; j < 4; j++) tempw[j] = w[j] + 0.5*k1[j];

			k2[0] = h*Function1(tempw, acritmin, acritmax);
			k2[1] = h*Function2(numelem, tempw);
			k2[2] = h*Function3(tempw, bcritmin, bcritmax);
			k2[3] = h*Function4(numelem, tempw);

			for (j = 0; j < 4; j++) tempw[j] = w[j] + k2[j];

			k3[0] = h*Function1(tempw, acritmin, acritmax);
			k3[1] = h*Function2(numelem, tempw);
			k3[2] = h*Function3(tempw, bcritmin, bcritmax);
			k3[3] = h*Function4(numelem, tempw);

			for (j = 0; j < 4; j++) w[j] += (k0[j] + 2*k1[j] + 2*k2[j] + k3[j])/6.0;
			
			Data[0][i] = w[0];
                        Data[1][i] = w[1];
			Data[2][i] = w[2];
                        Data[3][i] = w[3];                        

		}
		// set initial conditions to last w
		for (i = 0; i < 4; i++) initCond[i] = w[i];
	}
	

	private double Function1(double[] wstep, double rpmin, double rpmax) {
                double val;

                val=wstep[1];
                if (val > rpmax) val=rpmax;
                if (val < rpmin) val = rpmin;
                return val;
	}

	private double Function2(int elemnum, double[] wstep) {
		double first, second, third;
		first = -quadStrength[elemnum]*wstep[0];
		second = Math.pow(emittance,2)/Math.pow(wstep[0],3);
		third = 2*perveance/(wstep[0] + wstep[2]);
		return first + second + third;
	}

	private double Function3(double[] wstep, double rpmin, double rpmax) {
                double val;
		val=wstep[3];
                if (val > rpmax) val=rpmax;
                if (val < rpmin) val = rpmin;
                return val;
	}

	private double Function4(int elemnum, double[] wstep) {
		double first, second, third;
		first = quadStrength[elemnum]*wstep[2];
		second = Math.pow(emittance,2)/Math.pow(wstep[2],3);
		third = 2*perveance/(wstep[0] + wstep[2]);
		return first + second + third;
	}
	
	public double[][] getData() {
		return Data;
	}

	public double[] getRange() {
		return VecZ;
	}

	public double[] getFinalData() {
		double[] finalData = new double[4];
		int i;
		for (i = 0; i < 4; i++) finalData[i] = Data[i][VecZ.length-1];
		return finalData;
	}

	private double[] VecZ;
	private double[][] Data;
	private double aperture;
	private double emittance, perveance, energy, mass, initialperveance;
	private int numsteps;
	private double[] initCond = new double[4];
	private double[] quadStrength;
	private double[] stepsize;
	private int[] quadindices;
	private double neutralization;
	private int neutralizationindex;

}  // END Class BeamSolver


class GraphPanel extends JPanel implements Printable {

	public GraphPanel() {}

	public void Initialize (double[] DataX, double[][] DataY, Element[] lattice) {
	
		Dimension d = this.getPreferredSize();
		
		int n = DataX.length;
		float Ymax = (float) GetMax (DataY);
		float Xmax = (float) DataX[n-1];

		int numticksX = 8; // default 8 x tickmarks and 8 y
		int numticksY = 4; // for half the y axis
		int i;
	
		// get tickmark spacings
		float stepsizex = GetStep (Xmax/numticksX);
		float stepsizey = GetStep (Ymax/numticksY);
		
		numticksX = (int) Math.floor (Xmax/stepsizex);
                numticksY = (int) Math.ceil (Ymax/stepsizey);
                
		Ymax = numticksY*stepsizey;

		float height = d.height;
		float width = d.width;
		float ledge = 60;
		float yedge = 30;
		float redge = 40;
		float center = height/2;
		
		AffineTransform scaling = AffineTransform.getScaleInstance ((width-ledge-redge)/Xmax, -(height/2 - yedge)/Ymax);
		AffineTransform translation = AffineTransform.getTranslateInstance (ledge, center);
		
		xlabels = new TextLayout[numticksX];
		ylabels = new TextLayout[2*numticksY+1];
		
		axis = new GeneralPath();
		Font font = this.getFont();
		FontRenderContext frc = new FontRenderContext (new AffineTransform(), false, false);
                DecimalFormat df = new DecimalFormat ("0.##E0"); // was 0.## Peter Seidl
		for (i = 0; i < numticksX; i++) {
			float x = (i+1)*stepsizex;
                        axis.moveTo (x,0);
                        axis.lineTo (x,0);
                        xlabels[i] = new TextLayout (df.format(x), font, frc);
                }
		df = new DecimalFormat ("0.######"); // was 0.### Peter Seidl
                for (i = -numticksY; i <= numticksY; i++) {
                	float y = i*stepsizey;
                        axis.moveTo (0, y);
                        axis.lineTo (0, y);
			axis.moveTo (Xmax, y);
			axis.lineTo (Xmax, y);
                        ylabels[i+numticksY] = new TextLayout (df.format(y), font, frc);
                }
                axis.moveTo (0, -Ymax);
                axis.lineTo (0, +Ymax);
                axis.moveTo (Xmax, -Ymax);
                axis.lineTo (Xmax, +Ymax);
                axis.moveTo (0, 0);
                axis.lineTo (Xmax, 0);
                axis.transform (scaling);
                axis.transform (translation);

                int j;
                for (j = 0; j < 4; j += 2) {
                        curves[j] = new GeneralPath();
                        curves[j+1] = new GeneralPath();
			curves[j].moveTo ((float) DataX[0], (float)DataY[j][0]);
			curves[j+1].moveTo ((float)DataX[0], -(float)DataY[j][0]);
                        for (i = 1; i < n; i++) {
                        	curves[j].lineTo ((float)DataX[i], (float)DataY[j][i]);
                        	curves[j+1].lineTo ((float)DataX[i], -(float)DataY[j][i]);
                        }
                }
                
                for (j = 0; j < 4; j++) curves[j].transform (scaling);
		curves[4] = new GeneralPath();
		curves[5] = new GeneralPath();
		float x = 0;
		float y = height/2 - 6;
		if (lattice.length > 0 && lattice[0].getType().equals("quad")) y = height/2 - 27;
		curves[4].moveTo (0, y);
		curves[5].moveTo (0,-y);
		for (j = 0; j < lattice.length; j++) {
			if (lattice[j].getType().equals("drift")) y = height/2 - 6;
			if (lattice[j].getType().equals("quad")) y = height/2 - 27;
			if (lattice[j].getType().equals("drift") || lattice[j].getType().equals("quad")) {
				curves[4].lineTo (x, y);
				curves[5].lineTo (x,-y);
				x += lattice[j].getLength();
                                curves[4].lineTo (x, y);
                                curves[5].lineTo (x,-y);
			} else if (lattice[j].getType().equals("neutralization")) {
				for (y = height/2 - 27; y < height/2 - 6; y += 6) {
					curves[4].moveTo (x, y);
					curves[4].lineTo (x, y+3);
					curves[5].moveTo (x,-y);
					curves[5].lineTo (x,-y-3);
				}
			}
		}
		curves[4].transform (AffineTransform.getScaleInstance ((width-ledge-redge)/Xmax,1.));
		curves[5].transform (AffineTransform.getScaleInstance ((width-ledge-redge)/Xmax,1.));

	        for (j = 0; j < 6; j++) curves[j].transform (translation);
	}

	public double GetMax (double[][] data) {
   		// get max point in data set
   		// Note that data[1][j] and data[3][j] contain a' and b', so they are excluded.
		int j;
		double num = Math.abs(data[0][0]);
   		for (j = 0; j < data[0].length; j++) {
     			if (Math.abs(data[0][j]) > num) num = Math.abs(data[0][j]);
      			if (Math.abs(data[2][j]) > num) num = Math.abs(data[2][j]);
      		}
		return num;
	}

	private float GetStep (double step) {
		double[] bestValues = {2.0, 4.0, 5.0, 10.0}; 
		double scale = Math.pow (10, Math.floor (Math.log(step)/Math.log(10)));
		int index = 0;
		double min = Math.abs (step - bestValues[index]*scale);
		int i;
		for (i = 1; i < bestValues.length; i++) {
			double tempmin = Math.abs(step - bestValues[i]*scale);
			if (tempmin < min) {
				min = tempmin;
				index = i;
			}
		}
		return ((float) (bestValues[index]*scale));
	}

	public void showOptionsWindow() {
		String color = "Color";
		String bw = "Black and white";
		String initialoption;
		if (colorPrinting) initialoption = color;
		else initialoption = bw;
		Object[] options = {color, bw};
		Object selectedValue = JOptionPane.showInputDialog (null, "Print Color", "Printing options", JOptionPane.INFORMATION_MESSAGE, null, options, initialoption);
		if (color.equals(selectedValue)) colorPrinting = true;
		if (bw.equals(selectedValue)) colorPrinting = false;
	}

	public void showColorChooser() {
		colors[0] = JColorChooser.showDialog(this, "Horizontal beam radius", colors[0]);
		colors[1] = JColorChooser.showDialog(this, "Vertical beam radius", colors[1]);
		colors[2] = JColorChooser.showDialog(this, "Aperture", colors[2]);
		colors[3] = JColorChooser.showDialog(this, "Axes", colors[3]);
		colors[4] = JColorChooser.showDialog(this, "Background", colors[4]);
		repaint();
	}
		

	public void paintComponent (Graphics g) {
		super.paintComponent(g);
		Graphics2D g2 = (Graphics2D) g;
		setBackground(colors[4]);
		DrawPlot(g2, true);
	}

	public int print (Graphics g, PageFormat pf, int pi) throws PrinterException {
		if (pi >= 1) {
			return Printable.NO_SUCH_PAGE;
		}
		Graphics2D g2 = (Graphics2D) g;
		g2.setBackground(Color.white);
		g2.transform(AffineTransform.getTranslateInstance (pf.getImageableX(),pf.getImageableY()));
		Dimension d = this.getPreferredSize();
		g2.transform(AffineTransform.getScaleInstance (pf.getImageableWidth()/d.width, pf.getImageableHeight()/d.height));
		DrawPlot(g2, colorPrinting);
		return Printable.PAGE_EXISTS;
	}

        private void DrawPlot(Graphics2D g2, boolean color) {
                int i;
                // loop through number of equations
                g2.setPaint (Color.black);
                for (i = 0; i < 3; i ++) {
                        // draw each vector with different color
                        if (color) g2.setPaint (colors[i]);
			g2.draw(curves[2*i]);
			g2.draw(curves[2*i+1]);
                }
                if (color) g2.setPaint (colors[3]);
		g2.draw (axis);
                PathIterator pi = axis.getPathIterator(null);
		for (i = 0; i < xlabels.length; i++) {
                	float[] coords = new float[2];
                	pi.currentSegment (coords);
			g2.draw (new Line2D.Float (coords[0],coords[1]-4,coords[0],coords[1]+4));
			Rectangle2D bounds = xlabels[i].getBounds();
			float width = (float) bounds.getWidth();
			float height = (float) bounds.getHeight();
			xlabels[i].draw (g2, coords[0]-width/2,coords[1]+height+6);
                	pi.next();
                	pi.next();
                }
		for (i = 0; i < ylabels.length; i++) {
                	float[] coords = new float[2];
                	pi.currentSegment (coords);
			g2.draw (new Line2D.Float (coords[0]-4,coords[1],coords[0]+4,coords[1]));
			Rectangle2D bounds = ylabels[i].getBounds();
			float width = (float) bounds.getWidth();
			float height = (float) bounds.getHeight();
                	ylabels[i].draw (g2, coords[0]-width-6,coords[1]+height/2);
                	pi.next();
                	pi.next();
			pi.currentSegment (coords);
			g2.draw (new Line2D.Float (coords[0]-4,coords[1],coords[0]+4,coords[1]));
			pi.next();
			pi.next();
                }
        }

   private GeneralPath axis;
   private GeneralPath[] curves = new GeneralPath[6];
   private TextLayout[] xlabels;
   private TextLayout[] ylabels;
   private Color[] colors = {Color.green, new Color(180,180,255), Color.red, Color.yellow, Color.black};
   private boolean colorPrinting = false;
}
