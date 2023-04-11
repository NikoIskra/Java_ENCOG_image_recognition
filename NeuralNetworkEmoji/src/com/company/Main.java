package com.company;


import org.encog.*;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.train.strategy.ResetStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.platformspecific.j2se.TrainingDialog;
import org.encog.platformspecific.j2se.data.image.ImageMLData;
import org.encog.platformspecific.j2se.data.image.ImageMLDataSet;
import org.encog.util.downsample.Downsample;
import org.encog.util.downsample.RGBDownsample;
import org.encog.util.downsample.SimpleIntensityDownsample;
import org.encog.util.simple.EncogUtility;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;

public class Main {

    private int epohCount = 0;
    public static void main(final String args[]) {
        int trainingCount = new File("emojis/").listFiles().length;
        for(int i=0;i<trainingCount;i++){
            try {
                imageImport(i);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            Main program = new Main();
            program.execute("trening.txt");
        } catch (final Exception e) {
            e.printStackTrace();
        }
        Encog.getInstance().shutdown();
    }

    private static void imageImport(int index) throws IOException {
        boolean check = (new File("cuttedImages/br" + index + "/")).mkdirs();
        if (!check) {
            System.out.println("Datoteka br"+index+" vec postoji!");
        }else {
            String extension = ".jpg";
            File file = new File("emojis/br" + index + extension);
            BufferedImage image = ImageIO.read(file);
            int emojiWidth = image.getWidth() / 7;
            int emojiHeight = image.getHeight();
            int emojiCount = 7;
            BufferedImage emojis[] = new BufferedImage[emojiCount];
            int br = 0;
            for (int i = 0; i < emojiCount; i++) {

                emojis[br] = new BufferedImage(emojiWidth, emojiHeight, image.getType());
                Graphics2D graphics2D = emojis[br++].createGraphics();
                graphics2D.drawImage(image, 0, 0, emojiWidth, emojiHeight,
                        emojiWidth * i, 0,
                        emojiWidth * i + emojiWidth, emojiHeight, null);
                graphics2D.dispose();
            }
            System.out.println("Rezanje emojia je gotovo!");
            // Spremanje slika u specificne foldere za svaki font /br+index
            for (int i = 0; i < emojis.length; i++) {
                ImageIO.write(emojis[i], "jpg", new File("cuttedImages/br" + index + "/br" + i + ".jpg"));
            }
            System.out.println("Slike pod indexom br"+index+" su spremljene!");
        }
    }


    class ImagePair {
        private final File file;
        private final int identity;
        public ImagePair(final File file, final int identity) {
            super();
            this.file = file;
            this.identity = identity;
        }
        public File getFile() {
            return this.file;
        }
        public int getIdentity() {
            return this.identity;
        }
    }

    private final List<ImagePair> imageList = new ArrayList<ImagePair>();
    private final Map<String, String> args = new HashMap<String, String>();
    private final Map<String, Integer> identity2neuron = new HashMap<String, Integer>();
    private final Map<Integer, String> neuron2identity = new HashMap<Integer, String>();
    private ImageMLDataSet training;
    private String line;
    private int outputCount;
    private int downsampleWidth;
    private int downsampleHeight;
    private BasicNetwork network;

    private Downsample downsample;

    private int assignIdentity(final String identity) {
        if (this.identity2neuron.containsKey(identity.toLowerCase())) {
            return this.identity2neuron.get(identity.toLowerCase());
        }
        final int result = this.outputCount;
        this.identity2neuron.put(identity.toLowerCase(), result);
        this.neuron2identity.put(result, identity.toLowerCase());
        this.outputCount++;
        return result;
    }
    public void execute(final String file) throws IOException {
        final FileInputStream fstream = new FileInputStream(file);
        final DataInputStream in = new DataInputStream(fstream);
        final BufferedReader br = new BufferedReader(new InputStreamReader(in));
        while ((this.line = br.readLine()) != null) {
            executeLine();
        }
        in.close();
    }

    private void executeCommand(final String command, final Map<String, String> args) throws IOException {
        if (command.equals("input")) {
            processInput();
        } else if (command.equals("createtraining")) {
            processCreateTraining();
        } else if (command.equals("train")) {
            processTrain();
        } else if (command.equals("network")) {
            processNetwork();
        } else if (command.equals("whatis")) {
            processWhatIs();
        }
    }

    public void executeLine() throws IOException {
        final int index = this.line.indexOf(':');
        if (index == -1) {
            throw new EncogError("Invalid command: " + this.line);
        }
        final String command = this.line.substring(0, index).toLowerCase().trim();
        final String argsStr = this.line.substring(index + 1).trim();
        final StringTokenizer tok = new StringTokenizer(argsStr, ",");
        this.args.clear();
        while (tok.hasMoreTokens()) {
            final String arg = tok.nextToken();
            final int index2 = arg.indexOf(':');
            if (index2 == -1) {
                throw new EncogError("Invalid command: " + this.line);
            }
            final String key = arg.substring(0, index2).toLowerCase().trim();
            final String value = arg.substring(index2 + 1).trim();
            this.args.put(key, value);
        }

        executeCommand(command, this.args);
    }
    private String getArg(final String name) {
        final String result = this.args.get(name);
        if (result == null) {
            throw new EncogError("Missing argument " + name + " on line: "
                    + this.line);
        }
        return result;
    }
    private void processCreateTraining() {
        final String strWidth = getArg("width");
        final String strHeight = getArg("height");
        final String strType = getArg("type");

        this.downsampleHeight = Integer.parseInt(strHeight);
        this.downsampleWidth = Integer.parseInt(strWidth);

        if (strType.equals("RGB")) {
            this.downsample = new RGBDownsample();
        } else {
            this.downsample = new SimpleIntensityDownsample();
        }
        this.training = new ImageMLDataSet(this.downsample, false, 1, -1);
        System.out.println("Training set created");
    }

    private void processInput() throws IOException {
        final String image = getArg("image");
        final String identity = getArg("identity");

        final int idx = assignIdentity(identity);
        final File file = new File(image);

        this.imageList.add(new ImagePair(file, idx));

        System.out.println("Added input image:" + image);
    }

    private void processNetwork() throws IOException {
        System.out.println("Downsampling images...");

        for (final ImagePair pair : this.imageList) {
            final MLData ideal = new BasicMLData(this.outputCount);
            final int idx = pair.getIdentity();
            for (int i = 0; i < this.outputCount; i++) {
                if (i == idx) {
                    ideal.setData(i, 1);
                } else {
                    ideal.setData(i, -1);
                }
            }


            final Image img = ImageIO.read(pair.getFile());
            final ImageMLData data = new ImageMLData(img);
            this.training.add(data, ideal);
        }
        final String strHidden1 = getArg("hidden1");
        final String strHidden2 = getArg("hidden2");
        this.training.downsample(this.downsampleHeight, this.downsampleWidth);
        final int hidden1 = Integer.parseInt(strHidden1);
        final int hidden2 = Integer.parseInt(strHidden2);
        this.network = EncogUtility.simpleFeedForward(this.training.getInputSize(), hidden1, hidden2, this.training.getIdealSize(), true);
        System.out.println("Created network: " + this.network.toString());

    }

    private void processTrain() throws IOException {
        final String strMode = getArg("mode");
        final String strMinutes = getArg("minutes");
        final String strStrategyError = getArg("strategyerror");
        final String strStrategyCycles = getArg("strategycycles");

        System.out.println("Training Beginning... Output patterns="
                + this.outputCount);

        final double strategyError = Double.parseDouble(strStrategyError);
        final int strategyCycles = Integer.parseInt(strStrategyCycles);

//        final ManhattanPropagation train = new ManhattanPropagation(
//                this.network, this.training, strategyError);
//        final Backpropagation train = new Backpropagation(this.network, this.training);
//        train.addStrategy(new ResetStrategy(strategyError, strategyCycles));

        final ResilientPropagation train = new ResilientPropagation(this.network, this.training);
        train.addStrategy(new ResetStrategy(strategyError, strategyCycles));
        if (strMode.equalsIgnoreCase("gui")) {
            TrainingDialog.trainDialog(train, this.network, this.training);
        } else {
            final int minutes = Integer.parseInt(strMinutes);
            EncogUtility.trainConsole(train, this.network, this.training,
                    minutes);
        }
        System.out.println("Training Stopped...");
        epohCount += train.getIteration();
        System.out.println("Br epohe: " + epohCount);
    }


    public void processWhatIs() throws IOException {
        final String filename = getArg("image");
        final File file = new File(filename);
        BufferedImage[] emojis = importEmoji(file);
        String result = "";
        for (int i = 0; i < emojis.length; i++) {
            final Image img = emojis[i];
            ImageIO.write(emojis[i], "jpg", new File("cuttedImages/currentEmoji/" + "br" + i + ".jpg"));
            final ImageMLData input = new ImageMLData(img);
            input.downsample(this.downsample, false, this.downsampleHeight, this.downsampleWidth, 1, -1);
            final int winner = this.network.winner(input);
            result += neuron2identity.get(winner);
            result +="-";
        }
        System.out.println("Emoji s lokacije " + filename + ": " + result);
    }
    private static BufferedImage[] importEmoji(File curBroj) throws IOException {
        FileInputStream fis = new FileInputStream(curBroj);
        BufferedImage image = ImageIO.read(fis);
        int emojiCount = 7;
        int emojiWidth = image.getWidth() / 7;
        int emojiHeight = image.getHeight();
        int br = 0;
        BufferedImage emojis[] = new BufferedImage[emojiCount];
        for (int i = 0; i < 7; i++) {
            emojis[br] = new BufferedImage(emojiWidth, emojiHeight, image.getType());
            Graphics2D graphics2D = emojis[br++].createGraphics();
            graphics2D.drawImage(image, 0, 0, emojiWidth, emojiHeight,
                    emojiWidth * i, 0,
                    emojiWidth * i + emojiWidth, emojiHeight, null);
            graphics2D.dispose();
        }
        return emojis;
    }




}
