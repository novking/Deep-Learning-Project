import re
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET


class DrugBank:
    def half_life(self, element_text): #element.tag == "{http://www.drugbank.ca}half-life"
        if element_text:
            content = re.findall(
                r'(\d+)((?: hours| mins| days| day| hrs| min| h| minutes| hour|\) hour|\) hours|h| Hrs| weeks| week| secs| Hours| seconds|\+ hours|\) minutes| months))',
                i.text)
            if not content:
                print(i.text)
            for each_tuple_result in content:
                digit = float(each_tuple_result[0])
                unit = each_tuple_result[1]
                if unit == " days" or unit == " day":  # be careful, there's a space in front
                    final_result = digit
                elif unit == " mins" or unit == " min" or unit == " minutes" or unit == ") minutes":
                    final_result = digit / 60.0
                elif unit == " weeks" or unit == " week":
                    final_result = digit * 24 * 7
                elif unit == " secs" or unit == " seconds":
                    final_result = digit / 3600
                elif unit == " months":
                    final_result = digit * 24 * 30
                else:
                    final_result = digit * 24
        else:
            final_result = 0 # i think put some kinda average rom the if part will be good.
        return final_result # this is a float number. should I go with that?

    def protein_binding(self, element_text):
        pass

    def predicted_proerties(self, element_text):
        pass
