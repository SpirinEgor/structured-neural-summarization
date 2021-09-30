package java2graph;

import com.github.javaparser.JavaParser;
import com.github.javaparser.JavaToken;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.javadoc.Javadoc;
import com.github.javaparser.javadoc.JavadocBlockTag;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import me.tongfei.progressbar.ProgressBar;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;


public class Extractor {

    static String outputFile = "data";
    static String[] holdoutNames = {"train", "val", "test"};

    static Logger logger = Logger.getLogger("Java2Graph");

    public static void main(String[] args) throws IOException {
        InputStream loggingProperties = ClassLoader.getSystemResourceAsStream("logging.properties");
        LogManager.getLogManager().readConfiguration(loggingProperties);
        String datasetPath = args[0];
        String outputFolder = args[1];

        for (String holdoutName : holdoutNames) {
            processHoldout(datasetPath, holdoutName, outputFolder);
        }
    }

    private static void processHoldout(String datasetPath, String holdoutName, String outputFolder) throws IOException {
        // Set up a minimal type solver that only looks at the classes used to run this sample.
        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();
        combinedTypeSolver.add(new ReflectionTypeSolver());

        // Configure JavaParser to use type resolution
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(combinedTypeSolver);
        JavaParser.getStaticConfiguration().setSymbolResolver(symbolSolver);

        File outputPath = new File(outputFolder)
                .toPath().resolve(holdoutName).resolve(outputFile).toFile();
        outputPath.mkdirs();
        ChunkWriter<SerializableMethodData> writer = new ChunkWriter<>(outputPath.getAbsolutePath(), 5000);

        File holdoutFolder = new File(datasetPath).toPath().resolve(holdoutName).toFile();
        Collection<File> javaFiles = FileUtils.listFiles(holdoutFolder, new String[] {"java"}, true);

        for (File file: ProgressBar.wrap(javaFiles, holdoutName + " holdout")) {
            List<SerializableMethodData> allMethods = ExtractAllFromFile(file, combinedTypeSolver);
            allMethods.forEach(writer::add);
        }

        writer.close();
    }

    public static class SerializableMethodData {
        String Filename;
        String Span;
        Graph.JsonSerializableGraph Graph;
        String Signature;
        String Name;

        String Summary;
        String ReturnsSummary;
        Map<String, String> ParamsSummary;
    }

    public static List<SerializableMethodData> ExtractAllFromFile(File sourceFile, CombinedTypeSolver combinedTypeSolver) throws IOException {
        String fileContent = FileUtils.readFileToString(sourceFile, Charset.defaultCharset());
        CompilationUnit cu;
        try {
            cu = JavaParser.parse(fileContent);
        } catch (Exception e) {
            logger.log(Level.WARNING, "Failed to parse " + sourceFile, e);
            return Collections.emptyList();
        }

        List<SerializableMethodData> allMethods = new ArrayList<>();
        MethodFinder finder = new MethodFinder();
        cu.accept(finder, null);
        // Find all the calculations with two sides:
        for (MethodDeclaration decl : finder.allDeclarations) {
            if (!decl.getBody().isPresent()) continue;  // ignore interfaces
            Graph<Object> graph = JavaGraph.CreateGraph(decl, combinedTypeSolver);
            Map<Object, String> nodeLabelOverrides = new IdentityHashMap<>();
            nodeLabelOverrides.put(decl.getName().getTokenRange().get().getBegin(), "DECLARATION");
            Graph.JsonSerializableGraph jsonSerializableGraph = graph.toJsonSerializableObject(o -> NodePrinter(o, nodeLabelOverrides));

            SerializableMethodData methodData = new SerializableMethodData();
            methodData.Name = decl.getNameAsString();
            methodData.Signature = decl.getSignature() + "->" + decl.getTypeAsString();
            methodData.Filename = sourceFile.getAbsolutePath();

            if (decl.getRange().isPresent()) {
                methodData.Span = decl.getRange().get().toString();
            }

            if (decl.hasJavaDocComment()) {
                JavadocComment comment = decl.getJavadocComment().get();
                Javadoc javadoc = JavaParser.parseJavadoc(comment.toString());

                methodData.Summary = javadoc.getDescription().toText().replace("/**", "").trim();

                methodData.ParamsSummary = new HashMap<>();
                for (JavadocBlockTag blockTag : javadoc.getBlockTags()) {
                    String text = blockTag.getContent().toText().replace("/", "").trim();
                    if (blockTag.getTagName().equals("return")) {
                        methodData.ReturnsSummary = text;
                    } else if (blockTag.getTagName().equals("param")) {
                        if (blockTag.getName().isPresent()) {
                            methodData.ParamsSummary.put(blockTag.getName().get(), text);
                        }
                    }
                }
            }
            methodData.Graph = jsonSerializableGraph;

            allMethods.add(methodData);
        }
        return allMethods;
    }

    private static String NodePrinter(Object o, Map<Object, String> overrides) {
        if (overrides.containsKey(o)) {
            return overrides.get(o);
        }
        if (o instanceof JavaToken) {
            return ((JavaToken) o).getText();
        } else if (o instanceof Node) {
            return ((Node) o).getClass().getSimpleName();
        }
        throw new IllegalArgumentException("Unknown type of node.");
    }
}
