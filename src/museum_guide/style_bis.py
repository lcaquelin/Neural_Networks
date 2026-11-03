"""The 18 "styles bis": groups of artistically close styles (solution 1 of 08_analysis_and_perspectives).

Each of the 27 WikiArt styles belongs to exactly one style bis. The groups are used to read the
predictions of the 27-style models as 18 styles bis, to build an 18-class training set, and to tell
the visitor the story of the movement recognised by the guide.

Each style bis has:
    "description": the style bis itself. With several styles: one sentence, with its period and the names
        of its styles. With one style: two or three sentences, with its period, what characterises it
        and two or three major painters.
    "styles": {style: description}. With several styles, each style has one or two sentences: its period,
        what characterises it and two or three major painters. With one style, its description is None:
        the description of the style bis already describes it.

Sources: glossaries of Tate (art terms), The Metropolitan Museum of Art (Heilbrunn Timeline and essays),
National Galleries of Scotland, Encyclopaedia Britannica, and WikiArt for the content of its style labels.
"""

import numpy as np
import torch

STYLE_BIS = {
    "Renaissance": {
        "description": (
            "The Renaissance (about 1400-1600) gathers four styles of European painting: "
            "Early Renaissance, High Renaissance, Northern Renaissance and Mannerism."
        ),
        "styles": {
            "Early_Renaissance": (
                "Italy, about 1400-1490: painters rediscover linear perspective, anatomy and the models of "
                "antiquity to build believable spaces. Major painters: Masaccio, Fra Angelico, Sandro Botticelli."
            ),
            "High_Renaissance": (
                "Italy, about 1490-1527: the peak of the Renaissance, with balanced, harmonious and idealised "
                "compositions. Major painters: Leonardo da Vinci, Raphael, Michelangelo."
            ),
            "Northern_Renaissance": (
                "Flanders, the Netherlands and Germany, about 1430-1580: meticulous detail, rich textures and "
                "oil painting, for religious and everyday scenes. Major painters: Jan van Eyck, Albrecht Dürer, "
                "Pieter Bruegel the Elder."
            ),
            "Mannerism_Late_Renaissance": (
                "Italy then Europe, about 1520-1600: elongated figures, artificial poses, crowded compositions "
                "and acid colours, in reaction to the balance of the High Renaissance. Major painters: Pontormo, "
                "Parmigianino, Bronzino."
            ),
        },
    },
    "Baroque_Rococo": {
        "description": (
            "Baroque and Rococo (about 1600-1780) are two successive styles of European painting: "
            "Baroque and Rococo."
        ),
        "styles": {
            "Baroque": (
                "Europe, about 1600-1750: dramatic movement, strong contrasts of light and shadow (chiaroscuro) "
                "and intense emotion. Major painters: Caravaggio, Peter Paul Rubens, Rembrandt."
            ),
            "Rococo": (
                "France then Europe, about 1720-1780: light, playful and decorative scenes in pastel colours, "
                "often of love and aristocratic leisure. Major painters: Antoine Watteau, François Boucher, "
                "Jean-Honoré Fragonard."
            ),
        },
    },
    "Romanticism": {
        "description": (
            "Romanticism developed in Europe from about 1790 to 1850. Its painters favour emotion, imagination "
            "and the sublime power of nature over reason, in dramatic landscapes and scenes of history. "
            "Major painters: Caspar David Friedrich, J. M. W. Turner, Eugène Delacroix."
        ),
        "styles": {"Romanticism": None},
    },
    "Realism": {
        "description": (
            "Realism developed in France, then in the rest of Europe and in Russia, from about 1840 to 1880. "
            "Its painters depict ordinary people and everyday life truthfully, without idealisation, in reaction "
            "to Romanticism. Major painters: Gustave Courbet, Jean-François Millet, Ilya Repin."
        ),
        "styles": {"Realism": None},
    },
    # In WikiArt, "New Realism" means American Realism, not the French Nouveau Réalisme of the 1960s
    "New_Realism": {
        "description": (
            "New Realism, or American Realism, developed in the United States from about 1900 to 1960. "
            "Its painters depict modern American life, city streets, interiors and lonely figures, in a direct "
            "figurative style. Major painters: Edward Hopper, John Sloan, George Luks."
        ),
        "styles": {"New_Realism": None},
    },
    "Contemporary_Realism": {
        "description": (
            "Contemporary Realism developed mainly in North America from the 1960s to today. After decades of "
            "abstraction, its painters return to an objective, unidealised representation of people, objects "
            "and landscapes. Major painters: Andrew Wyeth, Philip Pearlstein, Alex Katz."
        ),
        "styles": {"Contemporary_Realism": None},
    },
    "Impressionism_PostImpressionism": {
        "description": (
            "Impressionism and Post-Impressionism (about 1860-1905) are two successive styles of French "
            "painting: Impressionism and Post-Impressionism."
        ),
        "styles": {
            "Impressionism": (
                "France, about 1860-1890: quick, visible brushstrokes and light colours catch the changing "
                "effects of light, often painted outdoors. Major painters: Claude Monet, Pierre-Auguste Renoir, "
                "Camille Pissarro."
            ),
            "Post_Impressionism": (
                "France, about 1885-1905: painters keep the bright colours of Impressionism but give more weight "
                "to structure, symbolic colour and personal expression. Major painters: Paul Cézanne, "
                "Vincent van Gogh, Paul Gauguin."
            ),
        },
    },
    "Pointillism": {
        "description": (
            "Pointillism, also called Neo-Impressionism or Divisionism, developed in France from about 1884 to "
            "1905. Its painters place small dots of pure colour side by side, so that they blend in the eye "
            "of the viewer, following the colour science of their time. Major painters: Georges Seurat, "
            "Paul Signac, Henri-Edmond Cross."
        ),
        "styles": {"Pointillism": None},
    },
    "Symbolism": {
        "description": (
            "Symbolism developed in Europe from about 1885 to 1910. Its painters express ideas, dreams, myths "
            "and mysteries rather than visible reality, often with strange or spiritual images. "
            "Major painters: Gustave Moreau, Odilon Redon, Arnold Böcklin."
        ),
        "styles": {"Symbolism": None},
    },
    "Art_Nouveau": {
        "description": (
            "Art Nouveau developed in Europe and the United States from about 1890 to 1910. It is a decorative "
            "style of flowing, curving lines inspired by plants and natural forms, used in painting, posters "
            "and illustration. Major artists: Alphonse Mucha, Gustav Klimt, Aubrey Beardsley."
        ),
        "styles": {"Art_Nouveau": None},
    },
    "Expressionism_Fauvism": {
        "description": (
            "Expressionism and Fauvism (about 1905-1925) are two movements of early modern painting that use "
            "bold colour to express emotion: Expressionism and Fauvism."
        ),
        "styles": {
            "Expressionism": (
                "Mainly Germany and Austria, about 1905-1925: distorted forms and violent colours express inner "
                "emotion and anxiety rather than outward reality. Major painters: Ernst Ludwig Kirchner, "
                "Egon Schiele, Edvard Munch (a forerunner)."
            ),
            "Fauvism": (
                "France, about 1905-1908: pure, intense colours applied in bold strokes, freed from the real "
                "colours of things. Major painters: Henri Matisse, André Derain, Maurice de Vlaminck."
            ),
        },
    },
    "Naive_Art": {
        "description": (
            "Naive Art, with Primitivism, developed from about 1885 to 1960, in Europe and then worldwide. "
            "Mostly self-taught artists paint with a childlike simplicity: flat colours, bold outlines, careful "
            "details and a free treatment of perspective. Major painters: Henri Rousseau, Niko Pirosmani, "
            "Grandma Moses."
        ),
        "styles": {"Naive_Art_Primitivism": None},
    },
    "Cubism": {
        "description": (
            "Cubism (about 1907-1925) gathers three styles: Cubism, Analytical Cubism and Synthetic Cubism."
        ),
        "styles": {
            "Cubism": (
                "France then Europe, about 1907-1925: objects are broken into geometric shapes and shown from "
                "several viewpoints at once, against traditional perspective. Major painters: Pablo Picasso, "
                "Georges Braque, Fernand Léger."
            ),
            "Analytical_Cubism": (
                "About 1908-1912: the first phase of Cubism, with objects split into small overlapping facets "
                "in a narrow palette of browns, greys and ochres. Major painters: Pablo Picasso, Georges Braque, "
                "Juan Gris."
            ),
            "Synthetic_Cubism": (
                "About 1912-1914: the second phase of Cubism, with simpler flat shapes, brighter colours and "
                "collages of paper and textures. Major painters: Pablo Picasso, Georges Braque, Juan Gris."
            ),
        },
    },
    "Abstract_Expressionism": {
        "description": (
            "Abstract Expressionism (about 1943-1960, mainly in New York) gathers two styles: "
            "Abstract Expressionism and Action Painting."
        ),
        "styles": {
            "Abstract_Expressionism": (
                "United States, mainly New York, about 1943-1960: large abstract canvases meant to express "
                "emotion, through gestures or through fields of colour. Major painters: Willem de Kooning, "
                "Arshile Gorky, Lee Krasner."
            ),
            "Action_painting": (
                "The gestural side of Abstract Expressionism, about 1947-1960: paint is poured, dripped or "
                "thrown in quick, spontaneous movements, the act of painting becoming the subject. "
                "Major painters: Jackson Pollock, Willem de Kooning, Franz Kline."
            ),
        },
    },
    "Color_Field_Painting": {
        "description": (
            "Color Field Painting developed in the United States, then in Britain, from about 1950 to 1970. "
            "Its painters cover large canvases with broad, flat areas of a few colours, for a calm and "
            "contemplative effect. Major painters: Mark Rothko, Barnett Newman, Helen Frankenthaler."
        ),
        "styles": {"Color_Field_Painting": None},
    },
    "Minimalism": {
        "description": (
            "Minimalism developed mainly in the United States from about 1960 to 1975. Its artists reduce the "
            "work to simple geometric forms, flat colour and repetition, without expression or illusion. "
            "Major painters: Frank Stella, Agnes Martin, Ellsworth Kelly."
        ),
        "styles": {"Minimalism": None},
    },
    "Pop_Art": {
        "description": (
            "Pop Art developed in Britain and the United States from about 1955 to 1970. Its artists take their "
            "images from mass culture (advertising, comics, consumer products, celebrities), often with flat "
            "colours and printing techniques. Major artists: Andy Warhol, Roy Lichtenstein, Richard Hamilton."
        ),
        "styles": {"Pop_Art": None},
    },
    "Ukiyo_e": {
        "description": (
            "Ukiyo-e, the \"pictures of the floating world\", developed in Japan from about 1670 to 1900, during "
            "the Edo period. They are mainly woodblock prints of actors, beautiful women, landscapes and legends, "
            "with flat colours and strong outlines. Major artists: Kitagawa Utamaro, Katsushika Hokusai, "
            "Utagawa Hiroshige."
        ),
        "styles": {"Ukiyo_e": None},
    },
}

# The class index of a style bis is its position in this list
STYLE_BIS_NAMES = list(STYLE_BIS)


def style_to_style_bis(style_names):
    """Return the index of the style bis of each style.

    Args:
        style_names: the 27 style names, in the order of the classes of the model
            (for example data["classes"] or class_names).

    Returns:
        A NumPy array of int64 with one style bis index per style, in the same order.
    """
    lookup = {style: g for g, group in enumerate(STYLE_BIS.values()) for style in group["styles"]}
    missing = [style for style in style_names if style not in lookup]
    if missing:
        raise ValueError(f"styles without a style bis: {missing}")
    return np.array([lookup[style] for style in style_names], dtype=np.int64)


def style_bis_probabilities(style_probs, style_names):
    """Turn probabilities over the styles into probabilities over the styles bis.

    The probability of a style bis is the sum of the probabilities of its styles,
    so the probabilities of the 18 styles bis still sum to 1.

    Args:
        style_probs: tensor of shape (N, 27), one probability per style (after softmax).
        style_names: the 27 style names, in the order of the columns of style_probs.

    Returns:
        A tensor of shape (N, 18), one probability per style bis.
    """
    mapping = torch.as_tensor(style_to_style_bis(style_names))
    style_probs = style_probs.cpu()
    bis_probs = torch.zeros(style_probs.shape[0], len(STYLE_BIS_NAMES), dtype=style_probs.dtype)
    # Add each style column to the column of its style bis
    bis_probs.index_add_(1, mapping, style_probs)
    return bis_probs