import re, json
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET


class DrugBank:
    def __init__(self):
        self.ATC_dict = json.loads('all_level.json')
        self.result = []
        for event, drug = ET.iterparse('drugbank.xml'):
            if drug.tag != '{http://www.drugbank.ca}drug':
                continue
            for first_level in drug:
                if first_level.tag == '{http://www.drugbank.ca}calculated-properties':
                    self.result += self.calculated_properties(first_level)

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

    def type(self,element_itself):
        content = element_itself.attrib
        return content['type']

    def ATC(self, *args):
        hot_vector_position = []
        for arg in args:
            position = self.ATC_dict[arg]
            hot_vector_position.append(position)
        return hot_vector_position


half_life = elem[15]
atc_codes = elem[32]
'{http://www.drugbank.ca}calculated-properties'
experimental_properties = elem[39]

level1 = atc_codes[0][1].attrib

x.get("atrib's name") = attrib's value
example:
name = country.get('name')
< country name = "Panama" >
    < rank > 68 < / rank >
    < year > 2011 < / year >
    < gdppc > 13600 < / gdppc >
    < neighbor name = "Costa Rica" direction = "W" / >
    < neighbor name = "Colombia" direction = "E" / >
< / country >

