/* Copyright (c) Dietmar Wolz.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory.
 */

package fcmaes.core;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Log {

    static class TeeStream extends PrintStream {
        PrintStream out;

        public TeeStream(PrintStream out1, PrintStream out2) {
            super(out1);
            this.out = out2;
        }

        public void write(byte buf[], int off, int len) {
            try {
                super.write(buf, off, len);
                out.write(buf, off, len);
            } catch (Exception e) {
            }
        }

        public void flush() {
            super.flush();
            out.flush();
        }
    }

    /**
     * Maps System.out and System.err both to the screen and into a log file.
     */

    public static void setLog() throws FileNotFoundException {

        String dateString = new SimpleDateFormat("yyyyMMddhhmm").format(new Date());
        File logFile = new File("log_" + dateString + ".log");
        @SuppressWarnings("resource")
        PrintStream logOut = new PrintStream(new FileOutputStream(logFile, true));

        PrintStream teeStdOut = new TeeStream(System.out, logOut);
        PrintStream teeStdErr = new TeeStream(System.err, logOut);

        System.setOut(teeStdOut);
        System.setErr(teeStdErr);
    }
}
